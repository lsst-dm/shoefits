from __future__ import annotations

__all__ = ("Schema", "FrameSchema")


import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar, cast

import pydantic
from jsonpointer import resolve_pointer
from pydantic.json_schema import JsonValue

from ._field import FieldInfo, _field_helper
from ._field_base import UnsupportedStructureError
from ._fits_schema import FitsExtensionSchema, FitsHeaderKeySchema, FitsSchemaConfiguration
from ._schema_path import Placeholders, SchemaPath
from .json_schema import JsonSchema


@dataclasses.dataclass
class FrameSchema:
    header_exports: dict[SchemaPath, FieldInfo] = dataclasses.field(default_factory=dict)
    data_exports: dict[SchemaPath, FieldInfo] = dataclasses.field(default_factory=dict)
    tree: dict[SchemaPath, str] = dataclasses.field(default_factory=dict)
    """Schema paths of nested frames.

    Keys are relative paths, values are keys in the parent
    `Schema.frame_schemas`.
    """


@dataclasses.dataclass
class Schema:
    json: JsonSchema
    frame_schemas: dict[str, FrameSchema] = dataclasses.field(default_factory=dict)
    tree: dict[SchemaPath, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def build(cls, frame_type: type[pydantic.BaseModel]) -> Schema:
        result = cls(json=cast(JsonSchema, frame_type.model_json_schema()))
        result._walk_json_schema(SchemaPath(), result.json, None, result.json["$defs"])
        return result

    def make_fits_schema(self, config: FitsSchemaConfiguration) -> list[FitsExtensionSchema]:
        return self._make_fits_schema(SchemaPath(), self.tree, [], config)

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
                tree[path] = nested_struct_name
                return
            deref = cast(JsonSchema, resolve_pointer(defs, nested_struct_name))
            raw_info.update(deref.pop("shoefits", {}))
            # MyPy is concerned that 'branch' and 'deref' might be different
            # members of the JsonSchema union; we know better.
            branch.update(deref)  # type: ignore[typeddict-item]
        if raw_info:
            info = _field_helper.type_adapter.validate_python(raw_info)
            if _field_helper.is_header_export(info):
                if frame is None:
                    raise UnsupportedStructureError(
                        f"Cannot export header information from {path} without an enclosing Frame."
                    )
                frame.header_exports[path] = info
            if _field_helper.is_data_export(info):
                if frame is None:
                    raise UnsupportedStructureError(
                        f"Cannot export data from {path} without an enclosing Frame."
                    )
                frame.data_exports[path] = info
            if not _field_helper.is_nested(info):
                # Exit early, since we know this field can't have anything
                # interesting below it.
                return
            elif nested_struct_name is not None:
                # This is a nested Frame; recurse into a new FrameSchema
                # starting from the already-dereferenced JSON Schema branch,
                # and then exit early so we don't also add its childen directly
                # to its parent.
                nested_frame = FrameSchema()
                self._walk_json_schema(SchemaPath(), branch, nested_frame, defs)
                self.frame_schemas[nested_struct_name] = nested_frame
                tree[path] = nested_struct_name
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

    def _make_fits_schema(
        self,
        path_prefix: SchemaPath,
        tree: dict[SchemaPath, str],
        parent_header: list[FitsHeaderKeySchema],
        config: FitsSchemaConfiguration,
    ) -> list[FitsExtensionSchema]:
        result: list[FitsExtensionSchema] = []
        for frame_path_suffix, frame_schema_name in tree.items():
            frame_path = path_prefix.join(frame_path_suffix)
            frame_schema = self.frame_schemas[frame_schema_name]
            common_header: list[FitsHeaderKeySchema] = parent_header.copy()
            for header_key_path_suffix, header_key_info in frame_schema.header_exports.items():
                common_header.extend(
                    _field_helper.generate_fits_header_schema(
                        header_key_path_suffix,
                        header_key_info,
                        config,
                    )
                )
            for data_path_suffix, data_info in frame_schema.data_exports.items():
                full_path = frame_path.join(data_path_suffix)
                header = common_header.copy()
                header.extend(_field_helper.generate_fits_header_schema(data_path_suffix, data_info, config))
                extension = FitsExtensionSchema(
                    label=config.get_extension_label(full_path),
                    frame_path=frame_path,
                    data_path=data_path_suffix,
                    data_info=data_info,
                    header=header,
                )
                result.append(extension)
            result.extend(self._make_fits_schema(frame_path, frame_schema.tree, common_header, config))
        return result
