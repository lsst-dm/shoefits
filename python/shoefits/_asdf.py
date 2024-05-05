from __future__ import annotations

__all__ = (
    "NdArray",
    "Quantity",
)

from typing import Literal

from ._dtypes import NumberType, Unit
from ._yaml import YamlModel


class NdArray(YamlModel, yaml_tag="!core/ndarray-1.1.0"):
    source: str
    shape: tuple[int, ...]
    datatype: NumberType
    byteorder: Literal["big"] = "big"


class Quantity(YamlModel, yaml_tag="!unit/quantity-1.2.0"):
    value: NdArray
    unit: Unit
