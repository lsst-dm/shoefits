from __future__ import annotations

__all__ = ("StructSchema", "DictSchema", "ListSchema", "CellGridSchema")


from typing import TypeAlias, Literal, Union, TYPE_CHECKING

from ._frames import FrameSchema, TractFrameSchema
from ._simple import SchemaBase

if TYPE_CHECKING:
    from ._all import Schema


class StructSchema(SchemaBase):
    schema_type: Literal["struct"] = "struct"
    fields: list[Schema]
    frame: FrameSchema | None = None


class DictSchema(SchemaBase):
    schema_type: Literal["dict"] = "dict"
    values: Schema
    frame: FrameSchema | None = None


class ListSchema(SchemaBase):
    schema_type: Literal["list"] = "list"
    values: Schema
    frame: FrameSchema | None = None


class CellGridSchema(SchemaBase):
    schema_type: Literal["cell_grid"] = "cell_grid"
    values: Schema
    frame: TractFrameSchema | None = None


_ContainerSchemaType: TypeAlias = Literal["struct", "dict", "list", "cell_grid"]
_ContainerSchema: TypeAlias = Union[StructSchema, DictSchema, ListSchema, CellGridSchema]
