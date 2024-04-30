from __future__ import annotations

__all__ = ("UnsupportedStructureError", "ValueFieldInfo", "make_value_field_info")

from typing import Literal, TypedDict

import numpy.typing as npt

from ._dtypes import Unit, ValueType, dtype_to_str


class UnsupportedStructureError(NotImplementedError):
    pass


class ValueFieldInfo(TypedDict):
    field_type: Literal["value"]
    dtype: ValueType
    unit: Unit | None
    fits_header: str | bool


def make_value_field_info(
    dtype: npt.DTypeLike,
    unit: Unit | None = None,
    fits_header: str | bool = False,
    field_type: Literal["value"] = "value",
) -> ValueFieldInfo:
    return ValueFieldInfo(
        dtype=dtype_to_str(dtype, ValueType), unit=unit, fits_header=fits_header, field_type=field_type
    )
