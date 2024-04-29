from __future__ import annotations

__all__ = ("Schema",)


import dataclasses
from collections.abc import Callable
from typing import ClassVar, cast, TypeAlias, Annotated, Union, Literal, Any

import pydantic
from jsonpointer import resolve_pointer
from pydantic.json_schema import JsonDict, JsonValue

from ._field_base import UnsupportedStructureError, ValueFieldInfo, FieldInfoBase
from ._schema_path import SchemaPath, PathPlaceholder
from ._field import FieldInfo, _FieldSchemaCallback


class Schema:
    def __init__(
        self,
        json: JsonDict,
        metadata: dict[str, FieldInfo],
        data: dict[str, FieldInfo],
        children: dict[SchemaPath, Schema],
    ):
        self.json = json
        self.metadata = metadata
        self.data = data
        self.children = children

    @classmethod
    def build(cls, frame_type: type[pydantic.BaseModel]) -> Schema:
        json_schema = frame_type.model_json_schema()
        result = cls(json_schema, {}, {}, {})
        for name, field_info in frame_type.model_fields.items():
            match field_info.json_schema_extra:
                case _FieldSchemaCallback(info=info):
                    if info.is_header_export:
                        result.metadata[name] = info
                    if info.is_data_export:
                        result.data[name] = info
        result._walk_json_schema(SchemaPath(), json_schema)
        return result

    def _walk_json_schema(self, path: SchemaPath, tree: JsonDict) -> None:
        if (pointer := tree.get("ref")) is not None:
            tree.update(resolve_pointer(self.json, cast(str, pointer).lstrip("#")))
            del tree["$ref"]
        if "$defs" in tree:
            # This branch holds the targets of JSON pointers, which we need to
            # keep in self.tree so we can dereference them, but we don't want
            # them in the local tree because that '$' won't unpack into kwargs.
            tree = tree.copy()
            del tree["$defs"]
        if (export_tree := tree.get("shoefits")) is not None:
            if not self._extract_export(cast(dict, export_tree), path):
                return
        try:
            schema_type = tree["type"]
        except KeyError:
            raise UnsupportedStructureError(f"Unsupported JSON Schema structure: {tree}.")
        if (walker := self._walkers.get(cast(str, schema_type))) is None:
            raise UnsupportedStructureError(f"Unsupported JSON Schema type: {schema_type!r}.")
        walker(self, path, **tree)

    def _walk_tree_object(
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
                self._walk_tree(path.push(k), v)
        elif additionalProperties is not None:
            # Mapping, with dynamic keys.
            self._walk_tree(path.push(PathPlaceholder.MAPPING), additionalProperties)
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
                self._walk_tree(path.push(index), item)
        elif items is not None:
            # List, with dynamic elements.
            self._walk_tree(path.push(PathPlaceholder.SEQUENCE), items)
        else:
            # Empty struct.
            pass

    def _walk_tree_scalar(self, **kwargs: JsonValue) -> None:
        pass

    def _extract_export(self, tree: JsonDict, path: SchemaPath) -> bool:
        if not path:
            raise UnsupportedStructureError(
                "Root object cannot be a FITS export; an outer struct is required."
            )
        export = _frame_field_helper.type_adapter.validate_python(tree)
        for root_path, branch_path in path.split_from_tail(1):
            if not root_path or not SchemaPath.is_term_dynamic(root_path.tail):
                break
        if (fits_ext := self.fits.get(root_path)) is not None:
            fits_ext = self.fits[root_path]
        else:
            fits_ext = FitsExtensionSchema()
            self.fits[root_path] = fits_ext
        if export.is_header_export:
            fits_ext.metadata[branch_path] = export
        if export.is_data_export:
            fits_ext.data[branch_path] = export
        return export.is_nested

    def _resolve_orphaned_metadata(self) -> None:
        orphans = {root_path: fits_ext for root_path, fits_ext in self.fits.items() if not fits_ext.data}
        for root_path in orphans:
            del self.fits[root_path]
        for root_path, old_fits_ext in orphans.items():
            for root_path, branch_prefix in root_path.split_from_tail(1):
                if (new_fits_ext := self.fits.get(root_path)) is not None:
                    for old_branch_path, metadata_key_schema in old_fits_ext.metadata.items():
                        new_branch_path = branch_prefix.join(old_branch_path)
                        new_fits_ext.metadata[new_branch_path] = metadata_key_schema
                    break
            else:
                raise UnsupportedStructureError(
                    f"Metadata keys starting at {branch_prefix} could not be associated "
                    "with FITS-exported data."
                )

    _walkers: ClassVar[dict[str, Callable[..., None]]] = {
        "object": _walk_tree_object,
        "array": _walk_tree_array,
        "string": _walk_tree_scalar,
        "integer": _walk_tree_scalar,
        "number": _walk_tree_scalar,
        "boolean": _walk_tree_scalar,
        "null": _walk_tree_scalar,
    }
