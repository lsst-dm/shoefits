from __future__ import annotations

__all__ = ()


import dataclasses
from collections.abc import Mapping
from typing import Self, cast

import astropy.io.fits
import numpy as np
import pydantic

from ._dtypes import BUILTIN_TYPES
from ._field_info import (
    FrameFieldInfo,
    HeaderFieldInfo,
    ImageFieldInfo,
    MappingFieldInfo,
    MaskFieldInfo,
    ModelFieldInfo,
    ValueFieldInfo,
)
from ._fits_schema import FitsExtensionLabelSchema, FitsExtensionSchema
from ._frame import Frame
from ._image import Image, ImageReference
from ._mask import Mask, MaskReference
from ._schema_path import Placeholders, SchemaPath
from ._yaml import YamlValue


@dataclasses.dataclass
class FitsExtensionLabel:
    extname: str
    extver: int | None
    extlevel: int

    @classmethod
    def build(
        cls,
        schema: FitsExtensionLabelSchema,
        frame_path: SchemaPath,
        frame_subs: Mapping[SchemaPath, str],
        subs_from_frame: Mapping[SchemaPath, str],
    ) -> Self:
        substitutions = dict(frame_subs)
        for path_from_frame, value in subs_from_frame.items():
            substitutions[frame_path.join(path_from_frame)] = value
        extname = schema.extname.format(substitutions).upper()
        extver = int(substitutions[schema.extver]) if schema.extver is not None else None
        return cls(extname=extname, extver=extver, extlevel=schema.extlevel)

    @property
    def asdf_source(self) -> str:
        result = f"fits:{self.extname}"
        if self.extver is not None:
            result = f"{result},{self.extver}"
        return result


@dataclasses.dataclass
class FitsExtensionWriter:
    schema: FitsExtensionSchema
    label: FitsExtensionLabel
    header: astropy.io.fits.Header
    data: np.ndarray


class FitsWriter:
    def __init__(self, schema: list[FitsExtensionSchema], frame: Frame):
        self.schema: dict[SchemaPath, FitsExtensionSchema] = {
            ext_schema.frame_path.join(ext_schema.data.path_from_frame): ext_schema for ext_schema in schema
        }
        self.writers: list[FitsExtensionWriter] = []
        self.tree = self._walk_frame(frame, SchemaPath(), {})

    def _walk_frame(
        self,
        frame: Frame,
        path: SchemaPath,
        frame_subs: Mapping[SchemaPath, str],
    ) -> YamlValue:
        result: dict[str, YamlValue] = {}
        for name, field_info in frame.frame_fields.items():
            value = getattr(frame, name)
            match field_info:
                case ValueFieldInfo():
                    result[name] = self._walk_value(value, field_info)
                case ImageFieldInfo():
                    result[name] = self._walk_image(cast(Image, value), path.push(name), frame_subs, {})
                case MaskFieldInfo():
                    result[name] = self._walk_mask(cast(Mask, value), path.push(name), frame_subs, {})
                case MappingFieldInfo():
                    result[name] = self._walk_mapping(
                        cast(Mapping[str, object], value),
                        field_info,
                        frame_path=path,
                        path_from_frame=SchemaPath(name, Placeholders.MAPPING),
                        frame_subs=frame_subs,
                        subs_from_frame={},
                    )
                case ModelFieldInfo():
                    context = {"yaml": True}
                    # TODO: add ASDF-block array handler to context
                    result[name] = cast(pydantic.BaseModel, value).model_dump(context=context)
                case HeaderFieldInfo():
                    # Header fields do not go in the YAML tree, and we pull
                    # them into the FITS header when we make the extensions.
                    pass
                case FrameFieldInfo():
                    result[name] = self._walk_frame(cast(Frame, value), path.push(name), frame_subs)
        # TODO: Wrap result in YamlDeferred to add a tag.
        return result

    def _walk_value(
        self,
        value: object,
        field_info: ValueFieldInfo,
    ) -> YamlValue:
        return BUILTIN_TYPES[field_info.dtype](value)

    def _walk_image(
        self,
        image: Image,
        full_path: SchemaPath,
        frame_subs: Mapping[SchemaPath, str],
        subs_from_frame: Mapping[SchemaPath, str],
    ) -> YamlValue:
        ext_schema = self.schema[full_path]
        label = FitsExtensionLabel.build(ext_schema.label, ext_schema.frame_path, frame_subs, subs_from_frame)
        ext_writer = FitsExtensionWriter(ext_schema, label, astropy.io.fits.Header(), image.array)
        self.writers.append(ext_writer)
        image_ref = ImageReference.from_image_and_source(image, label.asdf_source)
        return image_ref.model_dump(context={"yaml": True})

    def _walk_mask(
        self,
        mask: Mask,
        full_path: SchemaPath,
        frame_subs: Mapping[SchemaPath, str],
        subs_from_frame: Mapping[SchemaPath, str],
    ) -> YamlValue:
        ext_schema = self.schema[full_path]
        label = FitsExtensionLabel.build(ext_schema.label, ext_schema.frame_path, frame_subs, subs_from_frame)
        ext_writer = FitsExtensionWriter(ext_schema, label, astropy.io.fits.Header(), mask.array)
        self.writers.append(ext_writer)
        mask_ref = MaskReference.from_mask_and_source(mask, label.asdf_source)
        return mask_ref.model_dump(context={"yaml": True})

    def _walk_mapping(
        self,
        mapping: Mapping[str, object],
        field_info: MappingFieldInfo,
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        frame_subs: Mapping[SchemaPath, str],
        subs_from_frame: Mapping[SchemaPath, str],
    ) -> YamlValue:
        match field_info.value:
            case ValueFieldInfo():
                return {k: self._walk_value(v, field_info.value) for k, v in mapping.items()}
            case ImageFieldInfo():
                return {
                    k: self._walk_image(
                        cast(Image, v),
                        frame_path.join(path_from_frame),
                        frame_subs,
                        {path_from_frame: k, **subs_from_frame},
                    )
                    for k, v in mapping.items()
                }
            case MaskFieldInfo():
                return {
                    k: self._walk_mask(
                        cast(Mask, v),
                        frame_path.join(path_from_frame),
                        frame_subs,
                        {path_from_frame: k, **subs_from_frame},
                    )
                    for k, v in mapping.items()
                }
            case MappingFieldInfo():
                return {
                    k: self._walk_mapping(
                        cast(Mapping[str, object], v),
                        field_info.value,
                        frame_path,
                        path_from_frame.push(Placeholders.MAPPING),
                        frame_subs,
                        {path_from_frame: k, **subs_from_frame},
                    )
                    for k, v in mapping.items()
                }
            case ModelFieldInfo():
                context = {"yaml": True}
                # TODO: add ASDF-block array handler to context
                return {
                    k: cast(pydantic.BaseModel, v).model_dump(context=context) for k, v in mapping.items()
                }
            case HeaderFieldInfo():
                # Header fields do not go in the YAML tree, and we pull
                # them into the FITS header when we make the extensions.
                pass
            case FrameFieldInfo():
                return {
                    k: self._walk_frame(cast(Frame, v), frame_path.join(path_from_frame).push(k), frame_subs)
                    for k, v in mapping.items()
                }
        raise AssertionError()
