from __future__ import annotations

__all__ = ("Schema",)


import dataclasses
from collections.abc import Callable
from typing import ClassVar, cast, TypeAlias, Annotated, Union, Literal, Any

import pydantic
from jsonpointer import resolve_pointer
from pydantic.json_schema import JsonDict, JsonValue

from ._field_base import UnsupportedStructureError
from ._schema_path import SchemaPath, Placeholders
from ._field import FieldInfo, _FieldSchemaCallback, _field_helper


class Schema:
    def __init__(
        self,
        json: JsonDict,
        header_exports: dict[str, FieldInfo],
        data_exports: dict[str, FieldInfo],
        children: dict[SchemaPath, Schema],
    ):
        self.json = json
        self.header_exports = header_exports
        self.data_exports = data_exports
        self.children = children

    @classmethod
    def build(cls, frame_type: type[pydantic.BaseModel]) -> Schema:
        json_schema = frame_type.model_json_schema(mode="serialization")
        result = cls(json_schema, {}, {}, {})
        for name, field_info in frame_type.model_fields.items():
            match field_info.json_schema_extra:
                case _FieldSchemaCallback(info=info):
                    if _field_helper.is_header_export(info):
                        result.header_exports[name] = info
                    if _field_helper.is_data_export(info):
                        result.data_exports[name] = info
        result._walk_json_schema(SchemaPath(), json_schema)
        return result

    def _walk_json_schema(self, path: SchemaPath, tree: JsonDict) -> None:
        if (pointer := tree.get("ref")) is not None:
            deref = cast(dict, resolve_pointer(self.json, cast(str, pointer).lstrip("#")))
            if "shoefits" in deref:
                # Merge nested 'shoefits' dictionaries, instead of overriding
                # them completely.
                cast(dict, tree["shoefits"]).update(deref.pop("shoefits"))
            tree.update(deref)
            del tree["$ref"]
        if "$defs" in tree:
            # This branch holds the targets of JSON pointers, which we need to
            # keep in self.tree so we can dereference them, but we don't want
            # them in the local tree because that '$' won't unpack into kwargs.
            tree = tree.copy()
            del tree["$defs"]
        if (field_info := cast(FieldInfo | None, tree.get("shoefits"))) is not None:
            if _field_helper.is_frame(field_info):
                self.children
                return
        try:
            schema_type = tree["type"]
        except KeyError:
            raise UnsupportedStructureError(f"Unsupported JSON Schema structure: {tree}.")
        if (walker := self._walkers.get(cast(str, schema_type))) is None:
            raise UnsupportedStructureError(f"Unsupported JSON Schema type: {schema_type!r}.")
        walker(self, path, **tree)

    def _walk_json_schema_object(
        self,
        path: SchemaPath,
        properties: dict[str, JsonDict] | None = None,
        additionalProperties: JsonDict | None = None,
        **kwargs: JsonValue,
    ) -> None:
        if properties and additionalProperties is not None:
            raise UnsupportedStructureError(
                "Cannot handle 'object' tree with both 'properties' and 'additionalProperties'."
            )
        if properties:
            # Struct, with fixed fields.
            for k, v in properties.items():
                self._walk_json_schema(path.push(k), v)
        elif additionalProperties is not None:
            # Mapping, with dynamic keys.
            self._walk_json_schema(path.push(Placeholders.MAPPING), additionalProperties)
        else:
            # Empty struct.
            pass

    def _walk_tree_array(
        self,
        path: SchemaPath,
        items: JsonDict | None = None,
        prefixItems: list[JsonDict] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        if items is not None and prefixItems:
            raise UnsupportedStructureError("Cannot handle 'array' tree with both 'items' and 'prefixItems'.")
        if prefixItems:
            # Tuple, with fixed elements
            for index, item in enumerate(prefixItems):
                self._walk_json_schema(path.push(str(index)), item)
        elif items is not None:
            # List, with dynamic elements.
            self._walk_json_schema(path.push(Placeholders.SEQUENCE), items)
        else:
            # Empty struct.
            pass

    def _walk_tree_scalar(self, **kwargs: JsonValue) -> None:
        pass

    _walkers: ClassVar[dict[str, Callable[..., None]]] = {
        "object": _walk_json_schema_object,
        "array": _walk_tree_array,
        "string": _walk_tree_scalar,
        "integer": _walk_tree_scalar,
        "number": _walk_tree_scalar,
        "boolean": _walk_tree_scalar,
        "null": _walk_tree_scalar,
    }
