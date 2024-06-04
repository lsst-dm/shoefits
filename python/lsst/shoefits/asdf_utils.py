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
    "ArrayReader",
    "ArrayWriter",
    "InlineArrayModel",
    "ArrayReferenceModel",
    "ArrayModel",
    "ArraySerialization",
    "Array",
    "Quantity",
    "QuantityModel",
    "QuantitySerialization",
    "Time",
    "TimeModel",
    "TimeSerialization",
    "Unit",
    "UnitSerialization",
)

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Any, BinaryIO, Literal, TypeAlias, Union

import astropy.time
import astropy.units
import numpy as np
import pydantic
import pydantic_core.core_schema as pcs

from ._dtypes import NumberType, numpy_to_str, str_to_numpy


class ArrayWriter(ABC):
    """Serialization helper that gathers Numpy arrays from a Struct or
    Pydantic tree in order to write them elsewhere.
    """

    @abstractmethod
    def add_array(self, array: np.ndarray) -> str | int:
        raise NotImplementedError()


class BlockArrayWriter(ArrayWriter):
    """Implementation of ArrayWriter that writes to ASDF-style blocks just
    after the JSON tree.
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


class ArrayReader:
    """Deserialization helper that loads arrays that have been serialized
    outside the JSON tree.

    An instance of this class should be added to the Pydantic Validation
    Context dictionary with the "asdf_reader" key in order to allow nested
    objects to retreive their arrays.
    """

    def get_array(self, index: int | str) -> np.ndarray:
        raise NotImplementedError("TODO")


class UnitSerialization:
    """Pydantic hooks for unit serialization."""

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
    """Pydantic hooks for array serialization."""

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
    def from_model(cls, model: ArrayModel, info: pydantic.ValidationInfo) -> np.ndarray:
        match model:
            case ArrayReferenceModel(source=source):
                reader: ArrayReader
                if info.context is not None and (reader := info.context.get("asdf_reader")):
                    return reader.get_array(source)
                else:
                    raise ValueError(
                        "Serialized array is a reference, but no reader provided in validation context."
                    )
            case InlineArrayModel(data=data, datatype=datatype):
                dtype = str_to_numpy(datatype)
                return np.array(data, dtype=dtype)
        raise AssertionError("Unexpected member in ArrayModel union.")

    @classmethod
    def to_model(cls, array: np.ndarray, writer: ArrayWriter | None = None) -> ArrayModel:
        datatype = numpy_to_str(array.dtype, NumberType)
        if writer is None:
            return InlineArrayModel(data=array.tolist(), datatype=datatype)
        else:
            source = writer.add_array(array)
            return ArrayReferenceModel(source=source, shape=array.shape, datatype=datatype)

    @classmethod
    def serialize(
        cls,
        array: np.ndarray,
        info: pydantic.SerializationInfo,
    ) -> ArrayModel:
        writer: ArrayWriter | None = None
        if info is not None and info.context is not None:
            writer = info.context.get("array_writer")
        return cls.to_model(array, writer)


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
    def to_model(cls, quantity: astropy.units.Quantity, writer: ArrayWriter | None = None) -> QuantityModel:
        if quantity.isscalar:
            return QuantityModel(value=quantity.to_value(), unit=UnitSerialization.to_str(quantity.unit))
        else:
            return QuantityModel(
                value=ArraySerialization.to_model(quantity.to_value(), writer),
                unit=quantity.unit,
            )

    @classmethod
    def serialize(
        cls,
        quantity: astropy.units.Quantity,
        info: pydantic.SerializationInfo,
    ) -> QuantityModel:
        writer: ArrayWriter | None = None
        if info is not None and info.context is not None:
            writer = info.context.get("array_writer")
        return cls.to_model(quantity, writer)


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
    """Pydantic hooks for time serialization."""

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
        return TimeModel(value=time.to_value("fits"), scale=time.scale, format="iso")


Time: TypeAlias = Annotated[astropy.time.Time, TimeSerialization]
