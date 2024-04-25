from __future__ import annotations

__all__ = (
    "UnsignedIntegerType",
    "SignedIntegerType",
    "IntegerType",
    "FloatType",
    "Unit",
    "SchemaBase",
    "Sentinals",
    "UNKNOWN",
    "MULTIPLE",
    "MetadataKeySchema",
    "MetadataKeySchemaType",
    "NumberKeySchema",
    "StringKeySchema",
    "NumberType",
    "number_str_to_dtype",
    "dtype_to_number_str",
)

import enum
from typing import TypeAlias, Literal, Union, Annotated, get_args, cast

import pydantic
import numpy as np
import numpy.typing as npt

UnsignedIntegerType: TypeAlias = Literal["uint8", "uint16", "uint32", "uint64"]

SignedIntegerType: TypeAlias = Literal["int8", "int16", "int32", "int64"]

IntegerType: TypeAlias = SignedIntegerType | UnsignedIntegerType

FloatType: TypeAlias = Literal["float32", "float64"]

NumberType: TypeAlias = IntegerType | FloatType

Unit: TypeAlias = str


class Sentinals(enum.Enum):
    UNKNOWN = enum.auto()
    MULTIPLE = enum.auto()


UNKNOWN = Literal[Sentinals.UNKNOWN]
MULTIPLE = Literal[Sentinals.MULTIPLE]


class SchemaBase(pydantic.BaseModel):
    schema_type: str
    description: str | None = None

    @property
    def is_nested(self) -> bool:
        return False

    @property
    def is_metadata(self) -> bool:
        return False


class NumberKeySchema(SchemaBase):
    schema_type: Literal["number_key"] = "number_key"
    dtype: NumberType
    unit: Unit | None
    fits_key: str | None = None

    @property
    def is_metadata(self) -> bool:
        return True


class StringKeySchema(SchemaBase):
    schema_type: Literal["string_key"] = "string_key"
    size: int | None = None
    is_ascii: bool = False
    fits_key: str | None = None

    @property
    def is_metadata(self) -> bool:
        return True


MetadataKeySchemaType: TypeAlias = Literal["number_key", "string_key"]
_MetadataKeySchema: TypeAlias = Union[NumberKeySchema, StringKeySchema]
MetadataKeySchema: TypeAlias = Annotated[_MetadataKeySchema, pydantic.Field(discriminator="schema_type")]


def number_str_to_dtype(s: NumberType) -> npt.DTypeLike:
    return getattr(np, s)


def dtype_to_number_str(dtype: npt.DTypeLike) -> NumberType:
    result = np.dtype(dtype).name
    assert result in get_args(NumberType)
    return cast(NumberType, result)
