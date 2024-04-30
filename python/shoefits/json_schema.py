from __future__ import annotations

__all__ = (
    "JsonSchema",
    "ObjectJsonSchema",
    "ArrayJsonSchema",
    "IntegerJsonSchema",
    "NumberJsonSchema",
    "StringJsonSchema",
    "NullJsonSchema",
)


from typing import Any, Literal, TypeAlias, TypedDict, Union

JsonSchemaCommon = TypedDict(
    "JsonSchemaCommon",
    {
        "$ref": str,
        "$defs": dict[str, "JsonSchema"],
        "shoefits": dict[str, Any],
    },
    total=False,
)


class ObjectJsonSchema(JsonSchemaCommon, total=False):  # noqa: D101
    type: Literal["object"]
    properties: dict[str, JsonSchema]
    additionalProperties: list[JsonSchema]


class ArrayJsonSchema(JsonSchemaCommon, total=False):  # noqa: D101
    type: Literal["array"]
    items: JsonSchema
    prefixItems: list[JsonSchema]


class IntegerJsonSchema(JsonSchemaCommon, total=False):  # noqa: D101
    type: Literal["integer"]


class NumberJsonSchema(JsonSchemaCommon, total=False):  # noqa: D101
    type: Literal["number"]


class StringJsonSchema(JsonSchemaCommon, total=False):  # noqa: D101
    type: Literal["string"]


class NullJsonSchema(JsonSchemaCommon, total=False):  # noqa: D101
    type: Literal["null"]


JsonSchema: TypeAlias = Union[
    ObjectJsonSchema,
    ArrayJsonSchema,
    IntegerJsonSchema,
    NumberJsonSchema,
    StringJsonSchema,
    NullJsonSchema,
]
