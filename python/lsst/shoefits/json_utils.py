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
    "JsonValue",
    "JsonSchema",
    "ObjectJsonSchema",
    "ArrayJsonSchema",
    "IntegerJsonSchema",
    "NumberJsonSchema",
    "StringJsonSchema",
    "NullJsonSchema",
)


from typing import Any, Literal, TypeAlias, TypedDict, Union

JsonValue: TypeAlias = Union[int, str, float, None, list["JsonValue"], dict[str, "JsonValue"]]


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
