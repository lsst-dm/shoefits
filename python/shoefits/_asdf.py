from __future__ import annotations

__all__ = (
    "NdArray",
    "Quantity",
    "BlockWriter",
)

from io import BytesIO
from typing import Literal

import numpy as np

from ._dtypes import NumberType, Unit
from ._yaml import YamlModel


class NdArray(YamlModel, yaml_tag="!core/ndarray-1.1.0"):
    source: str | int
    shape: tuple[int, ...]
    datatype: NumberType
    byteorder: Literal["big"] = "big"


class Quantity(YamlModel, yaml_tag="!unit/quantity-1.2.0"):
    value: NdArray
    unit: Unit


class BlockWriter:
    def __init__(self) -> None:
        self._arrays: list[np.ndarray] = []

    def add_array(self, array: np.ndarray) -> int:
        source = len(self._arrays)
        self._arrays.append(array)
        return source

    def write(self, buffer: BytesIO) -> None:
        raise NotImplementedError("TODO")
