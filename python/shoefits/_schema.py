from __future__ import annotations

__all__ = ("Schema", "FrameSchema")


import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar, cast

import pydantic
from jsonpointer import resolve_pointer
from pydantic.json_schema import JsonValue

from ._field import (
    DataExportFieldInfo,
    HeaderExportFieldInfo,
    _field_helper,
    is_data_export,
    is_frame_field,
    is_header_export,
)
from ._field_base import UnsupportedStructureError
from ._schema_path import Placeholders, SchemaPath
from .json_schema import JsonSchema


@dataclasses.dataclass
class FrameSchema:
    name: str
    header_exports: dict[SchemaPath, HeaderExportFieldInfo] = dataclasses.field(default_factory=dict)
    data_exports: dict[SchemaPath, DataExportFieldInfo] = dataclasses.field(default_factory=dict)
    tree: dict[SchemaPath, FrameSchema] = dataclasses.field(default_factory=dict)
    """Schema paths of nested frames.

    Keys are relative paths, values are keys in the parent
    `Schema.frame_schemas`.
    """


@dataclasses.dataclass
class Schema:
    json: JsonSchema
    frame_schemas: dict[str, FrameSchema] = dataclasses.field(default_factory=dict)
    tree: dict[SchemaPath, FrameSchema] = dataclasses.field(default_factory=dict)

    @classmethod
    def build(cls, frame_type: type[pydantic.BaseModel]) -> Schema:
        result = cls(json=cast(JsonSchema, frame_type.model_json_schema()))
        result._walk_json_schema(SchemaPath(), result.json, None, result.json["$defs"])
        return result

    def _walk_json_schema(
        self,
        path: SchemaPath,
        branch: JsonSchema,
        frame: FrameSchema | None,
        defs: dict[str, JsonSchema],
    ) -> None:
        if frame is None:
            tree = self.tree
        else:
            tree = frame.tree
        # Extract the special shoefits subschema and remove it from the parent;
        # we don't need to duplicate the information it holds in the public
        # JSON schema.
        raw_info: dict[str, Any] = branch.pop("shoefits", {})
        # The JSON Schema for a nested Pydantic model is always a JSON Pointer
        # reference back to the $defs section.  Dereference that pointer and
        # merge it in with the field-level JSON Schema.
        nested_struct_name: str | None = None
        if "$ref" in branch:
            branch = branch.copy()
            pointer = branch.pop("$ref")
            nested_struct_name = pointer.removeprefix("#/$defs")
            if nested_struct_name in self.frame_schemas:
                tree[path] = self.frame_schemas[nested_struct_name]
                return
            deref = cast(JsonSchema, resolve_pointer(defs, nested_struct_name))
            raw_info.update(deref.pop("shoefits", {}))
            # MyPy is concerned that 'branch' and 'deref' might be different
            # members of the JsonSchema union; we know better.
            branch.update(deref)  # type: ignore[typeddict-item]
        if raw_info:
            info = _field_helper.type_adapter.validate_python(raw_info)
            if is_header_export(info):
                if frame is None:
                    raise UnsupportedStructureError(
                        f"Cannot export header information from {path} without an enclosing Frame."
                    )
                frame.header_exports[path] = info
            if is_data_export(info):
                if frame is None:
                    raise UnsupportedStructureError(
                        f"Cannot export data from {path} without an enclosing Frame."
                    )
                frame.data_exports[path] = info
            if not is_frame_field(info):
                # Exit early, since we know this field can't have anything
                # interesting below it.
                return
            elif nested_struct_name is not None:
                # This is a nested Frame; recurse into a new FrameSchema
                # starting from the already-dereferenced JSON Schema branch,
                # and then exit early so we don't also add its childen directly
                # to its parent.
                nested_frame = FrameSchema(nested_struct_name)
                self._walk_json_schema(SchemaPath(), branch, nested_frame, defs)
                self.frame_schemas[nested_struct_name] = nested_frame
                tree[path] = nested_frame
                return
        # Recurse into the JSON Schema to add associated nested exports with
        # this frame.
        try:
            schema_type = branch["type"]
        except KeyError:
            raise UnsupportedStructureError(f"Unsupported JSON Schema structure: {branch}.")
        # TODO: support `anyOf` etc. type unions, at least for None/null.
        if (walker := self._walkers.get(cast(str, schema_type))) is None:
            raise UnsupportedStructureError(f"Unsupported JSON Schema type: {schema_type!r}.")
        walker(self, path, frame, defs, **branch)

    def _walk_json_schema_object(
        self,
        path: SchemaPath,
        frame: FrameSchema,
        defs: dict[str, JsonSchema],
        *,
        properties: dict[str, JsonSchema] | None = None,
        additionalProperties: JsonSchema | None = None,
        **kwargs: JsonValue,
    ) -> None:
        if properties and additionalProperties is not None:
            raise UnsupportedStructureError(
                "Cannot handle 'object' tree with both 'properties' and 'additionalProperties'."
            )
        if properties:
            # Struct, with fixed fields.
            for k, v in properties.items():
                self._walk_json_schema(path.push(k), v, frame, defs)
        elif additionalProperties is not None:
            # Mapping, with dynamic keys.
            self._walk_json_schema(path.push(Placeholders.MAPPING), additionalProperties, frame, defs)
        else:
            # Empty struct.
            pass

    def _walk_tree_array(
        self,
        path: SchemaPath,
        frame: FrameSchema,
        defs: dict[str, JsonSchema],
        *,
        items: JsonSchema | None = None,
        prefixItems: list[JsonSchema] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        if items is not None and prefixItems:
            raise UnsupportedStructureError("Cannot handle 'array' tree with both 'items' and 'prefixItems'.")
        if prefixItems:
            # Tuple, with fixed elements
            for index, item in enumerate(prefixItems):
                self._walk_json_schema(path.push(str(index)), item, frame, defs)
        elif items is not None:
            # List, with dynamic elements.
            self._walk_json_schema(path.push(Placeholders.SEQUENCE), items, frame, defs)
        else:
            # Empty struct.
            pass

    def _walk_tree_scalar(
        self, path: SchemaPath, frame: FrameSchema, defs: dict[str, JsonSchema], **kwargs: JsonValue
    ) -> None:
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
