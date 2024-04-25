from __future__ import annotations

__all__ = (
    "FitsExportSchemaBase",
    "UnsupportedStructureError",
    "SchemaPathTerm",
    "UnionPath",
    "PathPlaceholder",
    "SchemaPath",
    "FitsExtensionLabelSchema",
)


import dataclasses
import enum
from collections.abc import Iterator
from typing import TypeAlias

import pydantic


class FitsExportSchemaBase(pydantic.BaseModel):
    export_type: str

    @property
    def is_nested(self) -> bool:
        return False

    @property
    def is_header_export(self) -> bool:
        return False

    @property
    def is_data_export(self) -> bool:
        return False


class UnsupportedStructureError(NotImplementedError):
    pass


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
