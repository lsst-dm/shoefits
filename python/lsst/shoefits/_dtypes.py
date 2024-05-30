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

__all__ = (
    "UnsignedIntegerType",
    "SignedIntegerType",
    "IntegerType",
    "FloatType",
    "NumberType",
    "NUMPY_TYPES",
    "str_to_numpy",
    "numpy_to_str",
    "ValueType",
)

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

NUMPY_TYPES: dict[str, type] = {k: getattr(np, k) for k in get_args(NumberType)}


def str_to_numpy(s: NumberType) -> type:
    return NUMPY_TYPES[s]


def numpy_to_str(dtype: npt.DTypeLike, kind: Any) -> Any:
    result = np.dtype(dtype).name
    if result not in get_args(kind):
        raise TypeError(f"Invalid dtype {result!r}; expected one of {get_args(kind)}.")
    return result


ValueType: TypeAlias = Literal["int", "str", "float", "bool"]
