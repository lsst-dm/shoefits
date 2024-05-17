from __future__ import annotations

__all__ = (
    "ValueFieldInfo",
    "ImageFieldInfo",
    "MaskFieldInfo",
    "StructFieldInfo",
    "MappingFieldInfo",
    "SequenceFieldInfo",
    "ModelFieldInfo",
    "HeaderFieldInfo",
    "FieldInfo",
)

import types
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union, cast, final, get_args, get_origin

import astropy.io.fits
import pydantic

from . import asdf_utils
from ._compression import FitsCompression
from ._dtypes import NumberType, UnsignedIntegerType, ValueType, numpy_to_str

if TYPE_CHECKING:
    from ._image import Image
    from ._mask import Mask, MaskPlane
    from ._struct import Struct


class FieldInfoBase(pydantic.BaseModel):
    description: str = ""
    allow_none: bool = False


@final
class ValueFieldInfo(FieldInfoBase):
    type_name: ValueType
    unit: asdf_utils.Unit | None = None
    fits_header: bool | str = False

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[object], **kwargs: Any
    ) -> ValueFieldInfo:
        if kwargs.setdefault("type_name", annotation.__name__) != annotation.__name__:
            raise ValueError(
                f"Annotation {annotation.__name__!r} and type_name={kwargs['type_name']!r} for "
                f"{struct_type.__name__}.{name} do not agree."
            )
        return cls.model_validate(kwargs)


@final
class ImageFieldInfo(FieldInfoBase):
    type_name: NumberType
    unit: asdf_utils.Unit | None = None
    fits_image_extension: bool | str = True

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[Image], **kwargs: Any
    ) -> ImageFieldInfo:
        if dtype_arg := kwargs.pop("dtype", None):
            dtype_type_name = numpy_to_str(dtype_arg, NumberType)
            if kwargs.setdefault("type_name", dtype_type_name) != dtype_type_name:
                raise ValueError(
                    f"dtype={dtype_arg!r} and type_name={kwargs['type_name']!r} are not consistent."
                )
        return ImageFieldInfo.model_validate(kwargs)


@final
class MaskFieldInfo(FieldInfoBase):
    type_name: UnsignedIntegerType = "uint8"
    required_planes: list[MaskPlane] = pydantic.Field(default_factory=list)
    allow_additional_planes: bool = True
    fits_image_extension: bool | str = True
    fits_plane_header_style: Literal["afw"] | None = "afw"
    fits_compression: FitsCompression | None = None

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[Mask], **kwargs: Any
    ) -> MaskFieldInfo:
        if dtype_arg := kwargs.pop("dtype", None):
            dtype_type_name = numpy_to_str(dtype_arg, NumberType)
            if kwargs.setdefault("type_name", dtype_type_name) != dtype_type_name:
                raise ValueError(
                    f"dtype={dtype_arg!r} and type_name={kwargs['type_name']!r} are not consistent."
                )
        return MaskFieldInfo.model_validate(kwargs)


@final
class StructFieldInfo(FieldInfoBase):
    cls: type[Struct]
    is_frame: bool

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[Struct], **kwargs: Any
    ) -> StructFieldInfo:
        from ._frame import Frame

        if not issubclass(kwargs.setdefault("cls", annotation), annotation):
            raise TypeError(
                f"Annotation {annotation.__name__} for frame field {struct_type.__name__}.{name} is not "
                f"consistent with cls={kwargs['cls']}."
            )
        kwargs["is_frame"] = issubclass(kwargs["cls"], Frame)
        return StructFieldInfo.model_validate(kwargs)


@final
class MappingFieldInfo(FieldInfoBase):
    cls: type[Mapping]
    value: FieldInfo

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        origin_type: type[Mapping[str, Any]],
        annotation: Any,
        **kwargs: Any,
    ) -> MappingFieldInfo:
        key_type, value_type = get_args(annotation)
        if key_type is not str:
            raise TypeError(
                f"Key type for mapping field {struct_type.__name__}.{name} must be 'str', not {key_type}."
            )
        return MappingFieldInfo(
            cls=origin_type,
            value=_build_field_info(struct_type, name, value_type, kwargs),
        )


@final
class SequenceFieldInfo(FieldInfoBase):
    cls: type[Sequence]
    value: FieldInfo

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        origin_type: type[Sequence[Any]],
        annotation: Any,
        **kwargs: Any,
    ) -> SequenceFieldInfo:
        value_type = get_args(annotation)
        return SequenceFieldInfo(
            cls=origin_type,
            value=_build_field_info(struct_type, name, value_type, kwargs),
        )


@final
class ModelFieldInfo(FieldInfoBase):
    cls: type[pydantic.BaseModel]

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[pydantic.BaseModel], **kwargs: Any
    ) -> ModelFieldInfo:
        if not issubclass(kwargs.setdefault("cls", annotation), annotation):
            raise TypeError(
                f"Annotation {annotation.__name__} for model field {struct_type.__name__}.{name} is not "
                f"consistent with cls={kwargs['cls']}."
            )
        return ModelFieldInfo.model_validate(kwargs)


@final
class HeaderFieldInfo(FieldInfoBase):
    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[astropy.io.fits.Header], **kwargs: Any
    ) -> HeaderFieldInfo:
        return HeaderFieldInfo.model_validate(kwargs)


FieldInfo: TypeAlias = Union[
    ValueFieldInfo,
    ImageFieldInfo,
    MaskFieldInfo,
    MappingFieldInfo,
    SequenceFieldInfo,
    ModelFieldInfo,
    HeaderFieldInfo,
    StructFieldInfo,
]


def _build_field_info(
    struct_type: type[Struct],
    name: str,
    annotation: Any,
    kwargs: dict[str, Any],
    annotation_had_none_stripped: bool = False,
) -> FieldInfo:
    from ._image import Image
    from ._mask import Mask
    from ._struct import Struct

    if isinstance(annotation, type):
        if kwargs.get("allow_none", False) and not annotation_had_none_stripped:
            raise TypeError(
                "Annotation does not permit None, but allow_none=True was set explicitly "
                f"on field {struct_type.__name__}.{name}."
            )
        if annotation is Image:
            return ImageFieldInfo.build(name, struct_type, annotation, **kwargs)
        if annotation is Mask:
            return MaskFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, Struct):
            return StructFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, pydantic.BaseModel):
            return ModelFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, astropy.io.fits.Header):
            return HeaderFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, Mapping):
            raise TypeError(f"Mapping field {struct_type.__name__}.{name!r} must have a type annotations.")
        return ValueFieldInfo.build(name, struct_type, annotation, **kwargs)
    origin_type = get_origin(annotation)
    if issubclass(origin_type, Mapping):
        return MappingFieldInfo.build(
            name, struct_type, cast(type[Mapping[str, Any]], origin_type), annotation, **kwargs
        )
    if issubclass(origin_type, Sequence):
        return SequenceFieldInfo.build(
            name, struct_type, cast(type[Sequence[Any]], origin_type), annotation, **kwargs
        )
    if origin_type is types.UnionType:
        match get_args(annotation):
            case (optional_type, types.NoneType) | (types.NoneType, optional_type):
                if not kwargs.setdefault("allow_none", True):
                    raise TypeError(
                        "Annotation permits None, but allow_none=False was set explicitly "
                        f"on field {struct_type.__name__}.{name}."
                    )
                return _build_field_info(
                    struct_type, name, optional_type, kwargs, annotation_had_none_stripped=True
                )
    raise TypeError(f"Unsupported type {annotation} for field {struct_type.__name__}.{name}.")
