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
)

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union, cast, final, get_args, get_origin

import astropy.io.fits
import pydantic

from ._asdf import Unit
from ._dtypes import (
    BUILTIN_TYPES,
    NUMPY_TYPES,
    NumberType,
    UnsignedIntegerType,
    ValueType,
    numpy_to_str,
)

if TYPE_CHECKING:
    from ._frame import Frame
    from ._image import Image
    from ._mask import Mask, MaskPlane
    from ._struct import Struct


class FieldInfoBase(pydantic.BaseModel):
    description: str = ""


@final
class ValueFieldInfo(FieldInfoBase):
    dtype: ValueType
    unit: Unit | None = None
    fits_header: bool | str = False

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[object], **kwargs: Any
    ) -> ValueFieldInfo:
        kwargs.setdefault("dtype", annotation)
        field_info = cls.model_validate(kwargs)
        if (
            BUILTIN_TYPES[field_info.dtype] is not annotation
            and NUMPY_TYPES[field_info.dtype] is not annotation
        ):
            raise TypeError(
                f"Annotation {annotation} for field {struct_type.__name__}.{name} is not consistent "
                f"with dtype={field_info.dtype!r}."
            )
        return field_info

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> ValueType:
        return numpy_to_str(v, ValueType)


@final
class ImageFieldInfo(FieldInfoBase):
    dtype: NumberType
    unit: Unit | None = None
    fits_image_extension: bool | str = True

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[Image], **kwargs: Any
    ) -> ImageFieldInfo:
        return ImageFieldInfo.model_validate(kwargs)

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> NumberType:
        return numpy_to_str(v, NumberType)


@final
class MaskFieldInfo(FieldInfoBase):
    dtype: UnsignedIntegerType
    required_planes: list[MaskPlane] = pydantic.Field(default_factory=list)
    allow_additional_planes: bool = True
    fits_image_extension: bool | str = True
    fits_plane_header_style: Literal["afw"] | None = "afw"

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[Mask], **kwargs: Any
    ) -> MaskFieldInfo:
        return MaskFieldInfo.model_validate(kwargs)

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> NumberType:
        return numpy_to_str(v, NumberType)


@final
class StructFieldInfo(FieldInfoBase):
    cls: type[Struct]
    is_frame: bool

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[Struct], **kwargs: Any
    ) -> StructFieldInfo:
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
    struct_type: type[Struct], name: str, annotation: Any, kwargs: dict[str, Any]
) -> FieldInfo:
    from ._image import Image
    from ._mask import Mask
    from ._struct import Struct

    if isinstance(annotation, type):
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
    raise TypeError(f"Unsupported type {annotation} for field {struct_type.__name__}.{name}.")
