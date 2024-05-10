from __future__ import annotations

__all__ = ("Frame", "Field", "Image")

from abc import ABC
from collections.abc import Mapping
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    get_type_hints,
)

from ._field_info import FieldInfo, _build_field_info
from ._image import Image

_T = TypeVar("_T")


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
                    frame_fields[name] = _build_field_info(cls, name, annotation, kwargs)
                except Exception as err:
                    raise TypeError(f"Error in definition for field {name!r}.") from err

        cls.frame_fields = frame_fields

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
