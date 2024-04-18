from __future__ import annotations

__all__ = (
    "UnsignedIntegerType",
    "SignedIntegerType",
    "IntegerType",
    "FloatType",
    "Unit",
    "SchemaBase",
    "IntegerSchema",
    "BoolSchema",
    "FloatSchema",
    "StringSchema",
    "ArraySchema",
    "Sentinals",
    "UNKNOWN",
)

import enum
from typing import TypeAlias, Literal, Union, TYPE_CHECKING, Self

import pydantic

if TYPE_CHECKING:
    from ._all import SchemaType

UnsignedIntegerType: TypeAlias = Literal["uint8", "uint16", "uint32", "uint64"]

SignedIntegerType: TypeAlias = Literal["int8", "int16", "int32", "int64"]

IntegerType: TypeAlias = SignedIntegerType | UnsignedIntegerType

FloatType: TypeAlias = Literal["float32", "float64"]

Unit: TypeAlias = str


class Sentinals(enum.Enum):
    UNKNOWN = enum.auto()

    @property
    def size(self) -> Self:
        return self

    @property
    def start(self) -> Self:
        return self


UNKNOWN = Literal[Sentinals.UNKNOWN]


class SchemaBase(pydantic.BaseModel):
    schema_type: SchemaType
    description: str | None = None


class IntegerSchema(SchemaBase):
    dtype: IntegerType
    unit: Unit | None = None


class BoolSchema(SchemaBase):
    pass


class FloatSchema(SchemaBase):
    dtype: FloatType
    unit: Unit | None = None


class StringSchema(SchemaBase):
    is_ascii_text: bool = False
    n_bytes_encoded: int | None = None


class ArraySchema(SchemaBase):
    dtype: IntegerType | FloatType
    nd: int | None = None
    shape: tuple[int, ...] | None = None
    unit: Unit | None = None


_SimpleSchemaType: TypeAlias = Literal["integer", "bool", "float", "string", "array"]
_SimpleSchema: TypeAlias = Union[IntegerSchema, BoolSchema, FloatSchema, StringSchema, ArraySchema]
