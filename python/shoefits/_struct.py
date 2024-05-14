from __future__ import annotations

__all__ = ("Struct", "Field")

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

_T = TypeVar("_T")


class Struct(ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        self._struct_data = {}

    def __init_subclass__(cls) -> None:
        try:
            annotations = get_type_hints(cls)
        except NameError as err:
            raise TypeError("Frames do not support forward type references or recursion.") from err
        kwargs: dict[str, Any]
        struct_fields: dict[str, FieldInfo] = {}
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
                    struct_fields[name] = _build_field_info(cls, name, annotation, kwargs)
                except Exception as err:
                    raise TypeError(f"Error in definition for field {name!r}.") from err

        cls.struct_fields = struct_fields

    _struct_data: dict[str, Any]
    struct_fields: ClassVar[Mapping[str, FieldInfo]]


class Field(Generic[_T]):
    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs

    def __get__(self, struct: Struct, struct_type: type[Struct] | None = None) -> _T:
        return struct._struct_data[self._name]

    def __set__(self, struct: Struct, value: _T) -> None:
        struct._struct_data[self._name] = value

    def __set_name__(self, owner: Struct, name: str) -> None:
        self._name = name
