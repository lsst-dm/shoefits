from __future__ import annotations

__all__ = ("ImageSchema", "MaskSchema", "MaskPixelType", "ArraySchema")


from typing import Literal, TypeAlias, Union, Annotated

import pydantic

from ._base import (
    Unit,
    UnsignedIntegerType,
    SchemaBase,
    UNKNOWN,
    Sentinals,
    NumberType,
)

MaskPixelType: TypeAlias = UnsignedIntegerType


class ArraySchema(SchemaBase):
    schema_type: Literal["array"] = "array"
    dtype: NumberType
    shape: tuple[int | UNKNOWN, ...] | UNKNOWN = Sentinals.UNKNOWN
    unit: Unit | None = None


class ImageSchema(SchemaBase):
    schema_type: Literal["image"] = "image"
    dtype: NumberType
    unit: Unit | None = None


class MaskSchema(SchemaBase):
    schema_type: Literal["mask"] = "mask"
    # {name: docs}; enumerate(planes) sets bit IDs.
    planes: dict[str, str] = pydantic.Field(default_factory=dict)
    dtype: UnsignedIntegerType = "uint8"


FitsDataExportSchemaType: TypeAlias = Literal["image", "mask", "array"]
_FitsDataExportSchema: TypeAlias = Union[ImageSchema, MaskSchema, ArraySchema]
FitsDataExportSchema: TypeAlias = Annotated[
    _FitsDataExportSchema, pydantic.Field(discriminator="schema_type")
]
