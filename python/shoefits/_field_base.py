from __future__ import annotations

__all__ = ("UnsupportedStructureError", "ValueFieldInfo", "make_value_field_info")

import re
from typing import Annotated, Any, Literal, TypeAlias, TypedDict

import numpy.typing as npt
import pydantic

from ._dtypes import Unit, ValueType, dtype_to_str

FITS_HEADER_RE = re.compile(r"[A-Z][A-Z0-9\-\_]{0-7}", re.ASCII)


class UnsupportedStructureError(NotImplementedError):
    pass


class _ValueFieldInfo(TypedDict):
    field_type: Literal["value"]
    dtype: ValueType
    unit: Unit | None
    fits_header: str | bool


def _validate(data: dict[str, Any]) -> dict[str, Any]:
    data["dtype"] = dtype_to_str(data["dtype"], ValueType)
    fits_header = data["fits_header"]
    if fits_header is not True and fits_header is not False and not FITS_HEADER_RE.fullmatch(fits_header):
        raise ValueError(
            f"Invalid FITS header key name: {fits_header!r}. "
            "Only uppercase letters, numbers, hyphens, and underscores are permitted. "
            "Must begin with an uppercase letter."
        )
    return data


ValueFieldInfo: TypeAlias = Annotated[_ValueFieldInfo, pydantic.BeforeValidator(_validate)]


def make_value_field_info(
    dtype: npt.DTypeLike,
    unit: Unit | None = None,
    fits_header: str | bool = False,
    field_type: Literal["value"] = "value",
) -> ValueFieldInfo:
    return ValueFieldInfo(
        dtype=dtype_to_str(dtype, ValueType), unit=unit, fits_header=fits_header, field_type=field_type
    )
