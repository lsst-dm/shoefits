from __future__ import annotations

__all__ = (
    "NdArray",
    "Quantity",
    "BlockWriter",
    "Unit",
)

from collections.abc import Iterable
from typing import Annotated, BinaryIO, Literal, TypeAlias

import astropy.units
import numpy as np
import pydantic

from ._dtypes import NumberType


class BlockWriter:
    """Serialization helper that gathers Numpy arrays from a Pydantic tree
    in order to write them later to ASDF blocks.
    """

    def __init__(self) -> None:
        self._arrays: list[np.ndarray] = []

    def add_array(self, array: np.ndarray) -> int:
        source = len(self._arrays)
        self._arrays.append(array)
        return source

    def write(self, buffer: BinaryIO) -> None:
        if self._arrays:
            raise NotImplementedError("TODO")

    def sizes(self) -> Iterable[int]:
        if self._arrays:
            raise NotImplementedError("TODO")
        return ()


def _deserialize_unit(value: object, handler: pydantic.ValidatorFunctionWrapHandler) -> astropy.units.Unit:
    if isinstance(value, astropy.units.Unit):
        return value
    string = handler(value)
    return astropy.units.Unit(string)


def _serialize_unit(unit: astropy.units.Unit) -> str:
    return unit.to_string("vounit")


Unit: TypeAlias = Annotated[
    astropy.units.Unit,
    pydantic.GetPydanticSchema(lambda _, h: h(str)),
    pydantic.WrapValidator(_deserialize_unit),
    pydantic.PlainSerializer(_serialize_unit),
    pydantic.WithJsonSchema(
        {
            "type": "string",
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/unit/unit-1.0.0",
            "tag": "!unit/unit-1.0.0",
        }
    ),
]


class NdArray(pydantic.BaseModel):
    """Model for the subset of the ASDF 'ndarray' schema used by shoefits."""

    source: str | int
    shape: tuple[int, ...]
    datatype: NumberType
    byteorder: Literal["big"] = "big"

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/core/ndarray-1.1.0",
            "tag": "!core/ndarray-1.1.0",
        }
    )


class Quantity(pydantic.BaseModel):
    """Model for the subset of the ASDF 'quantity' schema used by shoefits."""

    value: NdArray
    unit: Unit

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/unit/quantity-1.2.0",
            "tag": "!unit/quantity-1.2.0",
        }
    )
