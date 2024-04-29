from __future__ import annotations

__all__ = (
    "SchemaPathTerm",
    "SchemaPath",
)


from collections.abc import Iterator
from typing import TypeAlias, Any, ClassVar, Literal


from ._yaml import DeferredYaml


class Placeholders:
    MAPPING: ClassVar[Literal[":"]] = ":"
    SEQUENCE: ClassVar[Literal["*"]] = "*"


SchemaPathTerm: TypeAlias = str


class SchemaPath:
    def __init__(self, *args: SchemaPathTerm):
        self._terms = args

    @classmethod
    def from_str(cls, path: str) -> SchemaPath:
        return cls(*path.split("/"))

    def push(self, *terms: SchemaPathTerm) -> SchemaPath:
        return SchemaPath(*self._terms, *terms)

    def pop(self) -> tuple[SchemaPath, SchemaPathTerm]:
        return SchemaPath(*self._terms[:-1]), self._terms[-1]

    def join(self, other: SchemaPath) -> SchemaPath:
        return SchemaPath(*self._terms, *other._terms)

    def __str__(self) -> str:
        return "/".join(self._terms)

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

    def split_from_head(self, n: int = 0) -> Iterator[tuple[SchemaPath, SchemaPath]]:
        for i in range(n, len(self._terms)):
            yield SchemaPath(*self._terms[:i]), SchemaPath(*self._terms[i:])

    def split_from_tail(self, n: int = 0) -> Iterator[tuple[SchemaPath, SchemaPath]]:
        for i in reversed(range(len(self._terms) - n)):
            yield SchemaPath(*self._terms[:i]), SchemaPath(*self._terms[i:])

    @property
    def is_multiple(self) -> bool:
        return any(term == Placeholders.MAPPING or term == Placeholders.SEQUENCE for term in self._terms)

    @staticmethod
    def is_term_dynamic(term: SchemaPathTerm) -> bool:
        return term == Placeholders.MAPPING or term == Placeholders.SEQUENCE

    def resolve(self, tree: Any) -> Iterator[tuple[dict[SchemaPath, str | int], Any]]:
        return self._resolve(0, tree, {})

    def _resolve(
        self,
        depth: int,
        tree: Any,
        replacements: dict[SchemaPath, str | int],
    ) -> Iterator[tuple[dict[SchemaPath, str | int], Any]]:
        while depth < len(self._terms):
            if isinstance(tree, DeferredYaml):
                nested = tree.data
            else:
                nested = tree
            match self._terms[depth]:
                case Placeholders.MAPPING:
                    for k, v in nested.items():
                        yield from self._resolve(
                            depth + 1, v, replacements | {SchemaPath(*self._terms[: depth + 1]): k}
                        )
                case Placeholders.SEQUENCE:
                    for i, v in enumerate(nested):
                        yield from self._resolve(
                            depth + 1, v, replacements | {SchemaPath(*self._terms[: depth + 1]): i}
                        )
                case str() as key:
                    if key.isdigit():
                        tree = nested[int(key)]
                    else:
                        tree = nested[key]
                case _:
                    raise AssertionError()
            depth += 1
        yield tree
