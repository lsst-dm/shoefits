from __future__ import annotations

__all__ = (
    "UnsignedIntegerType",
    "SignedIntegerType",
    "IntegerType",
    "FloatType",
    "NumberType",
    "ValueType",
    "Unit",
    "str_to_dtype",
    "dtype_to_str",
)

from typing import TypeAlias, Literal, get_args, Any

import numpy as np
import numpy.typing as npt

UnsignedIntegerType: TypeAlias = Literal["uint8", "uint16", "uint32", "uint64"]

SignedIntegerType: TypeAlias = Literal["int8", "int16", "int32", "int64"]

IntegerType: TypeAlias = SignedIntegerType | UnsignedIntegerType

FloatType: TypeAlias = Literal["float32", "float64"]

NumberType: TypeAlias = IntegerType | FloatType

ValueType: TypeAlias = NumberType | Literal["str", "bytes"]

Unit: TypeAlias = str


def str_to_dtype(s: ValueType) -> npt.DTypeLike:
    return getattr(np, s)


def dtype_to_str(dtype: npt.DTypeLike, kind: Any) -> Any:
    result = np.dtype(dtype).name
    if result not in get_args(kind):
        raise TypeError(f"Invalid dtype {result!r}; expected one of {get_args(kind)}.")
    return result
