from __future__ import annotations

__all__ = ("ValueFieldInfo", "ImageFieldInfo", "Frame", "Field", "Image")

from abc import ABC
from collections.abc import Mapping
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeAlias,
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

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


class FieldInfoBase(pydantic.BaseModel):
    pass


@final
class ValueFieldInfo(FieldInfoBase):
    dtype: ValueType
    unit: Unit | None = None
    fits_header: bool | str = False

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> ValueType:
        return numpy_to_str(v, ValueType)


@final
class ImageFieldInfo(FieldInfoBase):
    dtype: NumberType
    unit: Unit | None = None
    fits_extension: bool | str | None = None

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> NumberType:
        return numpy_to_str(v, NumberType)


@final
class MaskFieldInfo(FieldInfoBase):
    dtype: UnsignedIntegerType
    required_planes: list[MaskPlane] = pydantic.Field(default_factory=list)
    allow_additional_planes: bool = True
    fits_extension: bool | str | None = None

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _accept_numpy(cls, v: Any) -> NumberType:
        return numpy_to_str(v, NumberType)


@final
class FrameFieldInfo(FieldInfoBase):
    cls: type[Frame]


@final
class MappingFieldInfo(FieldInfoBase):
    cls: type[Mapping]
    value: FieldInfo


@final
class ModelFieldInfo(FieldInfoBase):
    cls: type[pydantic.BaseModel]


@final
class HeaderFieldInfo(FieldInfoBase):
    pass


FieldInfo: TypeAlias = Union[
    ValueFieldInfo,
    ImageFieldInfo,
    MaskFieldInfo,
    MappingFieldInfo,
    ModelFieldInfo,
    HeaderFieldInfo,
    FrameFieldInfo,
]


class Frame(ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        self._frame_data = {}

    def __init_subclass__(cls) -> None:
        try:
            annotations = get_type_hints(cls)
        except NameError as err:
            raise TypeError("Frames do not support forward type references or recursion.") from err
        kwargs: dict[str, Any]
        frame_fields: dict[str, FieldInfo] = {}
        for name, attr in cls.__dict__.items():
            if isinstance(attr, Field):
                kwargs = attr._kwargs
                del attr._kwargs
                try:
                    annotation = annotations[name]
                except KeyError:
                    raise TypeError(
                        f"Frame field {cls.__name__}.{name} does not have a type annotation."
                    ) from None
                try:
                    frame_fields[name] = cls._resolve_field(name, annotation, kwargs)
                except Exception as err:
                    raise TypeError(f"Error in definition for field {name!r}.") from err
        cls.frame_fields = frame_fields

    @classmethod
    def _resolve_field(cls, name: str, annotation: Any, kwargs: dict[str, Any]) -> FieldInfo:
        if isinstance(annotation, type):
            if annotation is Image:
                return ImageFieldInfo.model_validate(kwargs)
            if annotation is Mask:
                return MaskFieldInfo.model_validate(kwargs)
            if issubclass(annotation, Frame):
                return FrameFieldInfo(cls=annotation)
            if issubclass(annotation, astropy.io.fits.Header):
                return HeaderFieldInfo()
            if issubclass(annotation, Mapping):
                raise TypeError(f"Mapping field {cls.__name__}.{name!r} must have a type annotations.")
            kwargs.setdefault("dtype", annotation)
            field_info = ValueFieldInfo.model_validate(kwargs)
            if (
                BUILTIN_TYPES[field_info.dtype] is not annotation
                and NUMPY_TYPES[field_info.dtype] is not annotation
            ):
                raise TypeError(
                    f"Annotation {annotation} for field {cls.__name__}.{name} is not consistent "
                    f"with dtype={field_info.dtype!r}."
                )
            return field_info
        origin_type = get_origin(annotation)
        if issubclass(origin_type, Mapping):
            key_type, value_type = get_args(annotation)
            if key_type is not str:
                raise TypeError(
                    f"Key type for mapping field {cls.__name__}.{name} must be 'str', not {key_type}."
                )
            return MappingFieldInfo(cls=origin_type, value=cls._resolve_field(name, value_type, kwargs))
        raise TypeError(f"Unsupported type {annotation.__name__} for field {cls.__name__}.{name}.")

    _frame_data: dict[str, Any]
    frame_fields: ClassVar[Mapping[str, FieldInfo]]


class Field(Generic[_T]):
    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs

    def __get__(self, frame: Frame, frame_type: type[Frame] | None = None) -> _T:
        return frame._frame_data[self._name]

    def __set__(self, frame: Frame, value: _T) -> None:
        frame._frame_data[self._name] = value

    def __set_name__(self, owner: Frame, name: str) -> None:
        self._name = name
