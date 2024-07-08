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

"""Pydantic models, validators, and serialization for conponents of the ASDF
schema that are used in this package.
"""

from __future__ import annotations

__all__ = (
    "InlineArrayModel",
    "ArrayReferenceModel",
    "ArrayModel",
    "ArraySerialization",
    "QuantityModel",
    "QuantitySerialization",
    "TimeModel",
    "TimeSerialization",
    "UnitSerialization",
    # Array, Quantity, Unit, and Time are lifted to package scope and should
    # not be imported directly from this module except from within the package.
)

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeAlias, Union

import astropy.time
import astropy.units
import numpy as np
import pydantic
import pydantic_core.core_schema as pcs

from ._dtypes import NumberType
from ._geom import Box
from ._read_context import ReadContext
from ._write_context import WriteContext


class UnitSerialization:
    """Pydantic hooks for unit serialization.

    This class provides implementations for the `Unit` type alias for
    `astropy.unit.Unit` that adds Pydantic serialization and validation.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_str_schema = pcs.chain_schema(
            [
                pcs.str_schema(),
                pcs.no_info_plain_validator_function(cls.from_str),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(astropy.units.UnitBase), from_str_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_str),
        )

    @classmethod
    def from_str(cls, value: str) -> astropy.units.UnitBase:
        return astropy.units.Unit(value, format="vounit")

    @staticmethod
    def to_str(unit: astropy.units.UnitBase) -> str:
        return unit.to_string("vounit")


Unit: TypeAlias = Annotated[
    astropy.units.UnitBase,
    UnitSerialization,
    pydantic.WithJsonSchema(
        {
            "type": "string",
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/unit/unit-1.0.0",
            "tag": "!unit/unit-1.0.0",
        }
    ),
]


class ArrayReferenceModel(pydantic.BaseModel):
    """Model for the subset of the ASDF 'ndarray' schema used by shoefits, in
    the case where the array data is stored elsewhere.
    """

    source: str | int
    shape: list[int]
    datatype: NumberType
    byteorder: Literal["big"] = "big"

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/core/ndarray-1.1.0",
            "tag": "!core/ndarray-1.1.0",
        }
    )


class InlineArrayModel(pydantic.BaseModel):
    """Model for the subset of the ASDF 'ndarray' schema used by shoefits, in
    the case where the array data is stored inline.
    """

    data: list[Any]
    datatype: NumberType

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/core/ndarray-1.1.0",
            "tag": "!core/ndarray-1.1.0",
        }
    )


def array_model_discriminator(obj: Any) -> str:
    """Discriminator function used to distinguish between `ArrayReferenceModel`
    and `InlineArrayModel` (in both Python and serialized forms).
    """
    if isinstance(obj, dict):
        return "reference" if "source" in obj else "inline"
    return "reference" if hasattr(obj, "source") else "inline"


ArrayModel: TypeAlias = Annotated[
    Union[
        Annotated[ArrayReferenceModel, pydantic.Tag("reference")],
        Annotated[InlineArrayModel, pydantic.Tag("inline")],
    ],
    pydantic.Discriminator(array_model_discriminator),
]


ArrayModelAdapter = pydantic.TypeAdapter(ArrayModel)


class ArraySerialization:
    """Pydantic hooks for array serialization.

    This class provides implementations for the `Array` type alias for
    `numpy.ndarray` that adds Pydantic serialization and validation.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                ArrayModelAdapter.core_schema,
                pcs.with_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(np.ndarray), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.serialize, info_arg=True),
        )

    @classmethod
    def from_model(
        cls,
        model: ArrayModel,
        info: pydantic.ValidationInfo,
        bbox_from_shape: Callable[[tuple[int, ...]], Box] = Box.from_shape,
        slice_result: Callable[[Box], tuple[slice, ...]] | None = None,
    ) -> np.ndarray:
        if read_context := ReadContext.from_info(info):
            if slice_result is None and (slice_bbox := read_context.get_parameter_bbox()) is not None:
                slice_result = slice_bbox.slice_within
            return read_context.get_array(model, bbox_from_shape, slice_result)
        match model:
            case ArrayReferenceModel():
                raise ValueError("Serialized array is a reference, but no read context provided.")
            case InlineArrayModel(data=data, datatype=datatype):
                return np.array(data, dtype=datatype.to_numpy())
        raise AssertionError("Unexpected member in ArrayModel union.")

    @classmethod
    def to_model(cls, array: np.ndarray, write_context: WriteContext | None = None) -> ArrayModel:
        datatype = NumberType.from_numpy(array.dtype)
        if write_context is None:
            return InlineArrayModel(data=array.tolist(), datatype=datatype)
        else:
            return write_context.add_array(array, None)

    @classmethod
    def serialize(
        cls,
        array: np.ndarray,
        info: pydantic.SerializationInfo,
    ) -> ArrayModel:
        return cls.to_model(array, WriteContext.from_info(info))


Array: TypeAlias = Annotated[np.ndarray, ArraySerialization]


class QuantityModel(pydantic.BaseModel):
    """Model for the subset of the ASDF 'quantity' schema used by shoefits."""

    value: ArrayModel | float
    unit: Unit

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/unit/quantity-1.2.0",
            "tag": "!unit/quantity-1.2.0",
        }
    )


class QuantitySerialization:
    """Pydantic hooks for quantity serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                QuantityModel.__pydantic_core_schema__,
                pcs.with_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema(
                [pcs.is_instance_schema(astropy.units.Quantity), from_model_schema]
            ),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_model, info_arg=True),
        )

    @classmethod
    def from_model(cls, model: QuantityModel, info: pydantic.ValidationInfo) -> np.ndarray:
        if isinstance(model.value, float):
            return astropy.units.Quantity(model.value, unit=model.unit)
        return astropy.units.Quantity(ArraySerialization.from_model(model.value, info), unit=model.unit)

    @classmethod
    def to_model(
        cls, quantity: astropy.units.Quantity, write_context: WriteContext | None = None
    ) -> QuantityModel:
        if quantity.isscalar:
            return QuantityModel(value=quantity.to_value(), unit=UnitSerialization.to_str(quantity.unit))
        else:
            return QuantityModel(
                value=ArraySerialization.to_model(quantity.to_value(), write_context),
                unit=quantity.unit,
            )

    @classmethod
    def serialize(
        cls,
        quantity: astropy.units.Quantity,
        info: pydantic.SerializationInfo,
    ) -> QuantityModel:
        return cls.to_model(quantity, WriteContext.from_info(info))


Quantity: TypeAlias = Annotated[astropy.units.Quantity, QuantitySerialization]


class TimeModel(pydantic.BaseModel):
    """Model for the subset of the ASDF 'time' schema used by shoefits."""

    value: str
    scale: Literal["utc", "tai"]
    format: Literal["iso"] = "iso"

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/time/time-1.2.0",
            "tag": "!time/time-1.2.0",
        }
    )


class TimeSerialization:
    """Pydantic hooks for time serialization.

    This class provides implementations for the `Time` type alias for
    `astropy.time.Time` that adds Pydantic serialization and validation.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                TimeModel.__pydantic_core_schema__,
                pcs.no_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(astropy.time.Time), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_model, info_arg=False),
        )

    @classmethod
    def from_model(cls, model: TimeModel) -> astropy.time.Time:
        return astropy.time.Time(model.value, scale=model.scale, format=model.format)

    @classmethod
    def to_model(cls, time: astropy.time.Time) -> TimeModel:
        if time.scale != "utc" and time.scale != "tai":
            time = time.tai
        return TimeModel(value=time.to_value("iso"), scale=time.scale, format="iso")


Time: TypeAlias = Annotated[astropy.time.Time, TimeSerialization]
