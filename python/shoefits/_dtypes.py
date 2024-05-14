from __future__ import annotations

__all__ = (
    "UnsignedIntegerType",
    "SignedIntegerType",
    "IntegerType",
    "FloatType",
    "NumberType",
    "ValueType",
    "NUMPY_TYPES",
    "BUILTIN_TYPES",
    "str_to_numpy",
    "numpy_to_str",
)

import builtins
from typing import Any, Literal, TypeAlias, get_args

import numpy as np
import numpy.typing as npt

# typing.get_args does not treat Union[Literal["a"], Literal["b"]] as
# equivalent to Literal["a", "b"], so we can't get away with defining any of
# the type aliases below in terms of each other without messing up our
# pydantic-style runtime evaluation of annotations elsewhere.

UnsignedIntegerType: TypeAlias = Literal["uint8", "uint16", "uint32", "uint64"]

SignedIntegerType: TypeAlias = Literal["int8", "int16", "int32", "int64"]

IntegerType: TypeAlias = Literal["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]

FloatType: TypeAlias = Literal["float32", "float64"]

NumberType: TypeAlias = Literal[
    "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"
]

ValueType: TypeAlias = Literal[
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "bool",
    "str",
    "bytes",
]

NUMPY_TYPES: dict[str, type] = {k: getattr(np, k) for k in get_args(NumberType)}
NUMPY_TYPES["bool"] = np.bool_
NUMPY_TYPES["str"] = np.str_
NUMPY_TYPES["bytes"] = np.bytes_
BUILTIN_TYPES: dict[str, type] = {k: getattr(builtins, k) for k in ["bool", "str", "bytes"]}
BUILTIN_TYPES.update(dict.fromkeys(get_args(IntegerType), int))
BUILTIN_TYPES.update(dict.fromkeys(get_args(FloatType), float))


def str_to_numpy(s: ValueType) -> type:
    return NUMPY_TYPES[s]


def numpy_to_str(dtype: npt.DTypeLike, kind: Any) -> Any:
    result = np.dtype(dtype).name
    if result not in get_args(kind):
        raise TypeError(f"Invalid dtype {result!r}; expected one of {get_args(kind)}.")
    return result
