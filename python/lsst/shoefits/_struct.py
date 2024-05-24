# This file is part of lsst-shoefits.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

__all__ = ("Struct", "Field", "field")

import dataclasses
from abc import ABC
from collections.abc import Mapping
from typing import Any, ClassVar, Self, dataclass_transform, get_type_hints

from ._field_info import FieldInfo, _build_field_info
from ._geom import Box


@dataclasses.dataclass
class ReadErrorDeferred:
    error: Exception


class Field:
    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs

    def __get__(self, struct: Struct | None, struct_type: type[Struct]) -> Any:
        if struct is None:
            return self
        result = struct._struct_data[self._name]
        if type(result) is ReadErrorDeferred:
            result.error.add_note(f"Trying to load attribute {struct_type.__name__}.{self._name}.")
            raise result.error
        return result

    def __set__(self, struct: Struct, value: Any) -> None:
        struct._struct_data[self._name] = value

    def __set_name__(self, owner: Struct, name: str) -> None:
        self._name = name


def field(**kwargs: Any) -> Any:
    return Field(**kwargs)


@dataclass_transform(eq_default=False, field_specifiers=(field,))
class Struct(ABC):
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        self = super().__new__(cls)
        self._struct_data = {}
        return self

    def __init__(self, bbox: Box | None = None, **kwargs: Any):
        for name, field_info in self.struct_fields.items():
            if name in kwargs:
                self._struct_data[name] = kwargs[name]
            else:
                self._struct_data[name] = field_info.get_default(type(self), name, bbox)

    @classmethod
    def from_box(cls, bbox: Box, **kwargs: Any) -> Self:
        return cls(bbox, **kwargs)

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
                struct_fields[name] = _build_field_info(cls, name, annotation, kwargs)
        cls.struct_fields = struct_fields

    @classmethod
    def _from_struct_data(cls, struct_data: dict[str, Any]) -> Self:
        result = cls.__new__(cls)
        result._struct_data.update(struct_data)
        return result

    _struct_data: dict[str, Any]
    struct_fields: ClassVar[Mapping[str, FieldInfo]]
