from __future__ import annotations

__all__ = (
    "ValueFieldInfo",
    "ImageFieldInfo",
    "MaskFieldInfo",
    "FrameFieldInfo",
    "MappingFieldInfo",
    "ModelFieldInfo",
    "HeaderFieldInfo",
)

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union, cast, final, get_args, get_origin

import astropy.io.fits
import pydantic

from ._dtypes import (
    BUILTIN_TYPES,
    NUMPY_TYPES,
    NumberType,
    Unit,
    UnsignedIntegerType,
    ValueType,
    numpy_to_str,
)
from ._image import Image
from ._mask import Mask, MaskPlane

_T = TypeVar("_T")

if TYPE_CHECKING:
    from ._frame import Frame


class FieldInfoBase(pydantic.BaseModel):
    pass


@final
class ValueFieldInfo(FieldInfoBase):
    dtype: ValueType
    unit: Unit | None = None
    fits_header: bool | str = False

    @classmethod
    def build(
        cls, name: str, frame_type: type[Frame], annotation: type[object], **kwargs: Any
    ) -> ValueFieldInfo:
        kwargs.setdefault("dtype", annotation)
        field_info = cls.model_validate(kwargs)
        if (
            BUILTIN_TYPES[field_info.dtype] is not annotation
            and NUMPY_TYPES[field_info.dtype] is not annotation
        ):
            raise TypeError(
                f"Annotation {annotation} for field {frame_type.__name__}.{name} is not consistent "
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
        cls, name: str, frame_type: type[Frame], annotation: type[Image], **kwargs: Any
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

    @classmethod
    def build(
        cls, name: str, frame_type: type[Frame], annotation: type[Mask], **kwargs: Any
    ) -> MaskFieldInfo:
        return MaskFieldInfo.model_validate(kwargs)

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> NumberType:
        return numpy_to_str(v, NumberType)


@final
class FrameFieldInfo(FieldInfoBase):
    cls: type[Frame]

    @classmethod
    def build(
        cls, name: str, frame_type: type[Frame], annotation: type[Frame], **kwargs: Any
    ) -> FrameFieldInfo:
        if not issubclass(kwargs.setdefault("cls", annotation), annotation):
            raise TypeError(
                f"Annotation {annotation.__name__} for frame field {frame_type.__name__}.{name} is not "
                f"consistent with cls={kwargs['cls']}."
            )
        return FrameFieldInfo.model_validate(kwargs)


@final
class MappingFieldInfo(FieldInfoBase):
    cls: type[Mapping]
    value: FieldInfo

    @classmethod
    def build(
        cls,
        name: str,
        frame_type: type[Frame],
        origin_type: type[Mapping[str, Any]],
        annotation: Any,
        **kwargs: Any,
    ) -> MappingFieldInfo:
        key_type, value_type = get_args(annotation)
        if key_type is not str:
            raise TypeError(
                f"Key type for mapping field {frame_type.__name__}.{name} must be 'str', not {key_type}."
            )
        return MappingFieldInfo(
            cls=origin_type,
            value=_build_field_info(frame_type, name, value_type, kwargs),
        )


@final
class ModelFieldInfo(FieldInfoBase):
    cls: type[pydantic.BaseModel]

    @classmethod
    def build(
        cls, name: str, frame_type: type[Frame], annotation: type[pydantic.BaseModel], **kwargs: Any
    ) -> ModelFieldInfo:
        if not issubclass(kwargs.setdefault("cls", annotation), annotation):
            raise TypeError(
                f"Annotation {annotation.__name__} for model field {frame_type.__name__}.{name} is not "
                f"consistent with cls={kwargs['cls']}."
            )
        return ModelFieldInfo.model_validate(kwargs)


@final
class HeaderFieldInfo(FieldInfoBase):
    @classmethod
    def build(
        cls, name: str, frame_type: type[Frame], annotation: type[astropy.io.fits.Header], **kwargs: Any
    ) -> HeaderFieldInfo:
        return HeaderFieldInfo.model_validate(kwargs)


FieldInfo: TypeAlias = Union[
    ValueFieldInfo,
    ImageFieldInfo,
    MaskFieldInfo,
    MappingFieldInfo,
    ModelFieldInfo,
    HeaderFieldInfo,
    FrameFieldInfo,
]


def _build_field_info(
    frame_type: type[Frame], name: str, annotation: Any, kwargs: dict[str, Any]
) -> FieldInfo:
    # TODO: refactor at least some of this into FieldInfo classmethods.
    if isinstance(annotation, type):
        if annotation is Image:
            return ImageFieldInfo.build(name, frame_type, annotation, **kwargs)
        if annotation is Mask:
            return MaskFieldInfo.build(name, frame_type, annotation, **kwargs)
        if issubclass(annotation, Frame):
            return FrameFieldInfo.build(name, frame_type, annotation, **kwargs)
        if issubclass(annotation, pydantic.BaseModel):
            return ModelFieldInfo.build(name, frame_type, annotation, **kwargs)
        if issubclass(annotation, astropy.io.fits.Header):
            return HeaderFieldInfo.build(name, frame_type, annotation, **kwargs)
        if issubclass(annotation, Mapping):
            raise TypeError(f"Mapping field {frame_type.__name__}.{name!r} must have a type annotations.")
        return ValueFieldInfo.build(name, frame_type, annotation, **kwargs)
    origin_type = get_origin(annotation)
    if issubclass(origin_type, Mapping):
        return MappingFieldInfo.build(
            name, frame_type, cast(type[Mapping[str, Any]], origin_type), annotation, **kwargs
        )
    raise TypeError(f"Unsupported type {annotation} for field {frame_type.__name__}.{name}.")
