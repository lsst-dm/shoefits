from __future__ import annotations

__all__ = ("Schema",)


import dataclasses
import enum
from collections.abc import Callable, Iterator
from typing import TypeAlias, Union, Annotated, ClassVar, cast

import pydantic
from jsonpointer import resolve_pointer
from pydantic.json_schema import JsonDict, JsonValue

from ._base import _MetadataKeySchema, MetadataKeySchema
from ._arrays import _FitsDataExportSchema, FitsDataExportSchema


class UnsupportedStructureError(NotImplementedError):
    pass


FitsExportSchema: TypeAlias = Annotated[
    Union[_FitsDataExportSchema, _MetadataKeySchema], pydantic.Field(discriminator="schema_type")
]


class PathPlaceholder(enum.Enum):
    MAPPING = ":"
    SEQUENCE = "*"

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class UnionPath:
    n: int

    def __str__(self) -> str:
        return f"?{self.n}"


SchemaPathTerm: TypeAlias = PathPlaceholder | UnionPath | str | int


class SchemaPath:
    def __init__(self, *args: SchemaPathTerm):
        self._terms = args

    def push(self, *terms: SchemaPathTerm) -> SchemaPath:
        return SchemaPath(*self._terms, *terms)

    def pop(self) -> tuple[SchemaPath, SchemaPathTerm]:
        return SchemaPath(*self._terms[:-1]), self._terms[-1]

    def join(self, other: SchemaPath) -> SchemaPath:
        return SchemaPath(*self._terms, *other._terms)

    def __str__(self) -> str:
        return "/".join(map(str, self._terms))

    def __hash__(self) -> int:
        return hash(self._terms)

    def __eq__(self, value: object) -> bool:
        return self._terms == getattr(value, "_terms")

    def __iter__(self) -> Iterator[SchemaPathTerm]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self)

    @property
    def head(self) -> SchemaPathTerm:
        return self._terms[0]

    @property
    def tail(self) -> SchemaPathTerm:
        return self._terms[-1]

    def split_from_back(self, n: int = 0) -> Iterator[tuple[SchemaPath, SchemaPath]]:
        for i in reversed(range(len(self) - n)):
            yield SchemaPath(*self._terms[:i]), SchemaPath(*self._terms[i:])

    @property
    def is_union(self) -> bool:
        return any(type(term) is UnionPath for term in self._terms)

    @property
    def is_multiple(self) -> bool:
        return any(
            term is PathPlaceholder.MAPPING or term is PathPlaceholder.SEQUENCE for term in self._terms
        )

    @staticmethod
    def is_term_dynamic(term: SchemaPathTerm) -> bool:
        return term is PathPlaceholder.MAPPING or term is PathPlaceholder.SEQUENCE or type(term) is UnionPath


@dataclasses.dataclass
class FitsExtensionLabelSchema:
    @classmethod
    def from_schema_path(cls, path: SchemaPath) -> FitsExtensionLabelSchema:
        result = cls(extname="")
        cumulative: list[SchemaPathTerm] = []
        # First pass: just look for sequences; we'll use the last one's index
        # for EXTVER.
        for term in path:
            if term is PathPlaceholder.SEQUENCE:
                result.extver = SchemaPath(*cumulative)
            cumulative.append(term)
        # Second pass: process all terms to build extname and placeholders.
        cumulative.clear()
        extname_terms: list[str] = []
        for term in path:
            match term:
                case str():
                    extname_terms.append(term)
                case int():
                    extname_terms.append(str(term))
                case PathPlaceholder():
                    placeholder_path = SchemaPath(*cumulative)
                    if placeholder_path != result.extver:
                        extname_terms.append(f"{{{len(result.placeholders)}}}")
                        result.placeholders.append(placeholder_path)
                case UnionPath():
                    # Unions shouldn't have terms in user-visible paths, even
                    # though we have to track them internally.  We just
                    # remember that this makes the HDU optional.
                    result.optional = True
                case _:
                    raise AssertionError(term)
            cumulative.append(term)
        result.extname = "/".join(extname_terms)
        return result

    extname: str
    """A string placeholder used to form the EXTNAME header value.

    This will include positional `str.format` placeholders for each entry in
    `placeholders`, to be filled by the actual mapping key or sequence index
    when writing a file.
    """

    placeholders: list[SchemaPath] = dataclasses.field(default_factory=list)
    """Paths to dynamic mappings and sequences whose names or indices need
    to substituted into `extname`.
    """

    extver: SchemaPath | None = None
    """Path to a single dynamic mapping whose index should be used for the
    EXTVER header.
    """

    optional: bool = False
    """If `True`, this FITS extension may not exist in all files with this
    schema.
    """


@dataclasses.dataclass
class FitsExtensionSchema:
    metadata: dict[SchemaPath, MetadataKeySchema] = dataclasses.field(default_factory=dict)
    data: dict[SchemaPath, FitsDataExportSchema] = dataclasses.field(default_factory=dict)


class Schema:
    def __init__(self, tree: JsonDict):
        self.tree: JsonDict
        self.fits: dict[SchemaPath, FitsExtensionSchema] = {}
        self._walk_tree(SchemaPath(), tree)
        self._resolve_orphaned_metadata()

    def _walk_tree(self, path: SchemaPath, tree: JsonDict) -> None:
        if (pointer := tree.get("ref")) is not None:
            self.tree.update(resolve_pointer(cast(str, pointer).lstrip("#")))
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
        schema_type = tree["type"]
        if isinstance(schema_type, list):
            for n, union_type in enumerate(schema_type):
                tree_copy = tree.copy()
                tree_copy["type"] = union_type
                self._walk_tree(path.push(UnionPath(n)), tree_copy)
        else:
            self._walkers[cast(str, schema_type)](self, path, **tree)

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
        export = self._export_type_adapter.validate_python(tree)
        for root_path, branch_path in path.split_from_back(1):
            if not root_path or not SchemaPath.is_term_dynamic(root_path.tail):
                break
        if (fits_ext := self.fits.get(root_path)) is not None:
            fits_ext = self.fits[root_path]
        else:
            fits_ext = FitsExtensionSchema()
            self.fits[root_path] = fits_ext
        if export.is_metadata:
            fits_ext.metadata[branch_path] = cast(MetadataKeySchema, export)
        else:
            fits_ext.data[branch_path] = cast(FitsDataExportSchema, export)
        return export.is_nested

    def _resolve_orphaned_metadata(self) -> None:
        orphans = {root_path: fits_ext for root_path, fits_ext in self.fits.items() if not fits_ext.data}
        for root_path in orphans:
            del self.fits[root_path]
        for root_path, old_fits_ext in orphans.items():
            for root_path, branch_prefix in root_path.split_from_back(1):
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

    _export_type_adapter: ClassVar[pydantic.TypeAdapter[FitsExportSchema]] = pydantic.TypeAdapter(
        FitsExportSchema
    )
