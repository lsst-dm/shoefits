from __future__ import annotations

__all__ = (
    "FitsHeaderSchema",
    "FitsValueHeaderSchema",
    "FitsImageHeaderSchema",
    "FitsDataSchema",
    "FitsImageDataSchema",
    "FitsExtensionLabelSchema",
    "FitsExtensionSchema",
)


import dataclasses
from abc import ABC
from collections.abc import Sequence

from ._field_info import (
    FieldInfo,
    HeaderFieldInfo,
    ImageFieldInfo,
    MappingFieldInfo,
    MaskFieldInfo,
    SequenceFieldInfo,
    StructFieldInfo,
    ValueFieldInfo,
)
from ._frame import Frame
from ._schema_path import Placeholders, SchemaPath, SchemaPathName


@dataclasses.dataclass
class FitsHeaderSchema:
    path_from_frame: SchemaPath


@dataclasses.dataclass
class FitsValueHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS header cards exported from a simple `str`,
    `int`, or `float` value in the YAML tree.
    """

    field_info: ValueFieldInfo
    key: SchemaPathName


@dataclasses.dataclass
class FitsOpaqueHeaderSchema(FitsHeaderSchema):
    """Schema definition for a bundle of FITS opaque header cards that do not
    appear in the YAML tree at all, and are typically just propagated from
    some externally-written FITS file.
    """

    field_info: HeaderFieldInfo


@dataclasses.dataclass
class FitsImageHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS headers exported by all `Image` objects."""

    field_info: ImageFieldInfo


@dataclasses.dataclass
class FitsMaskHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS headers exported by all `Mask` objects."""

    field_info: MaskFieldInfo


@dataclasses.dataclass
class FitsDataSchema:
    path_from_frame: SchemaPath
    """Path of the object that maps to this FITS extension, relative to
    `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsImageDataSchema(FitsDataSchema):
    field_info: ImageFieldInfo


@dataclasses.dataclass
class FitsMaskDataSchema(FitsDataSchema):
    field_info: MaskFieldInfo


@dataclasses.dataclass
class FitsExtensionLabelSchema:
    """Schema for the identifier header columns in a FITS extension."""

    extname: SchemaPathName
    extlevel: int
    extver: SchemaPath | None = None


@dataclasses.dataclass
class FitsExtensionSchema:
    label: FitsExtensionLabelSchema
    frame_path: SchemaPath
    header: list[FitsHeaderSchema]
    data: FitsDataSchema


class FitsSchemaConfiguration(ABC):
    def build_fits_schema(self, frame_type: type[Frame]) -> list[FitsExtensionSchema]:
        return self._walk_frame(frame_type, [], SchemaPath(), 1)

    def _walk_frame(
        self,
        frame_type: type[Frame],
        parent_header: Sequence[FitsHeaderSchema],
        path: SchemaPath,
        extlevel: int,
    ) -> list[FitsExtensionSchema]:
        frame_header = list(parent_header)
        for field_name, field_info in frame_type.struct_fields.items():
            if (
                header_schema := self._extract_frame_header_schema(field_info, SchemaPath(field_name))
            ) is not None:
                frame_header.append(header_schema)
        results: list[FitsExtensionSchema] = []
        for field_name, field_info in frame_type.struct_fields.items():
            results.extend(
                self._extract_extension_schema(
                    field_info, frame_header, path, SchemaPath(field_name), extlevel
                )
            )
        return results

    def _extract_frame_header_schema(
        self,
        field_info: FieldInfo,
        path_from_frame: SchemaPath,
    ) -> FitsHeaderSchema | None:
        match field_info:
            case ValueFieldInfo():
                return self.get_value_header_schema(path_from_frame, field_info)
            case HeaderFieldInfo():
                return FitsOpaqueHeaderSchema(path_from_frame, field_info)
            case MappingFieldInfo():
                return self._extract_frame_header_schema(
                    field_info.value, path_from_frame.push(Placeholders.MAPPING)
                )
        return None

    def _extract_extension_schema(
        self,
        field_info: FieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        extlevel: int,
    ) -> list[FitsExtensionSchema]:
        results: list[FitsExtensionSchema] = []
        match field_info:
            case ImageFieldInfo():
                if (
                    extension_schema := self.get_image_extension_schema(
                        field_info,
                        frame_header,
                        frame_path=frame_path,
                        path_from_frame=path_from_frame,
                        extlevel=extlevel,
                    )
                ) is not None:
                    results.append(extension_schema)
            case MaskFieldInfo():
                if (
                    extension_schema := self.get_mask_extension_schema(
                        field_info,
                        frame_header,
                        frame_path=frame_path,
                        path_from_frame=path_from_frame,
                        extlevel=extlevel,
                    )
                ) is not None:
                    results.append(extension_schema)
            case MappingFieldInfo():
                results.extend(
                    self.get_mapping_extensions_schema(
                        field_info,
                        frame_header,
                        frame_path=frame_path,
                        path_from_frame=path_from_frame,
                        extlevel=extlevel,
                    )
                )
            case SequenceFieldInfo():
                raise NotImplementedError("TODO")
            case StructFieldInfo():
                raise NotImplementedError("TODO")
        return results

    def get_value_header_schema(
        self, path_from_frame: SchemaPath, field_info: ValueFieldInfo
    ) -> FitsValueHeaderSchema | None:
        if field_info.fits_header is True:
            template_terms: list[str] = []
            substitutions: list[SchemaPath] = []
            for term in path_from_frame:
                match term:
                    case Placeholders.MAPPING | Placeholders.SEQUENCE:
                        template_terms.append(f"{{{len(substitutions)}}}")
                        substitutions.append(SchemaPath(*path_from_frame))
                    case _:
                        template_terms.append(term.upper())
            key = SchemaPathName(template="-".join(template_terms), substitutions=tuple(substitutions))
        elif field_info.fits_header:
            key = SchemaPathName(template=field_info.fits_header)
        else:
            return None
        return FitsValueHeaderSchema(path_from_frame, field_info, key)

    def get_extension_label_schema(
        self, path_from_frame: SchemaPath, extlevel: int
    ) -> FitsExtensionLabelSchema:
        raise NotImplementedError("TODO")

    def get_image_extension_schema(
        self,
        field_info: ImageFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        extlevel: int,
    ) -> FitsExtensionSchema | None:
        if not field_info.fits_image_extension:
            return None
        elif field_info.fits_image_extension is not True:  # i.e. it's a string
            label = FitsExtensionLabelSchema(
                extname=SchemaPathName(field_info.fits_image_extension), extlevel=extlevel
            )
        else:
            label = self.get_extension_label_schema(frame_path.join(path_from_frame), extlevel)
        header = list(frame_header)
        header.append(FitsImageHeaderSchema(path_from_frame, field_info))
        return FitsExtensionSchema(
            label,
            frame_path,
            header,
            FitsImageDataSchema(path_from_frame, field_info),
        )

    def get_mask_extension_schema(
        self,
        field_info: MaskFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        extlevel: int,
    ) -> FitsExtensionSchema | None:
        if not field_info.fits_image_extension:
            return None
        elif field_info.fits_image_extension is not True:  # i.e. it's a string
            label = FitsExtensionLabelSchema(
                extname=SchemaPathName(field_info.fits_image_extension), extlevel=extlevel
            )
        else:
            label = self.get_extension_label_schema(frame_path.join(path_from_frame), extlevel)
        header = list(frame_header)
        header.append(FitsMaskHeaderSchema(path_from_frame, field_info))
        return FitsExtensionSchema(
            label,
            frame_path,
            header,
            FitsMaskDataSchema(path_from_frame, field_info),
        )

    def get_mapping_extensions_schema(
        self,
        field_info: MappingFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        extlevel: int,
    ) -> list[FitsExtensionSchema]:
        return self._extract_extension_schema(
            field_info.value,
            frame_header,
            frame_path,
            path_from_frame.push(Placeholders.MAPPING),
            extlevel,
        )


# TODO:
# - override methods for all FieldInfo fits_* options.
