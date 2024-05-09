from __future__ import annotations

__all__ = (
    "SchemaPathTerm",
    "SchemaPath",
    "SchemaPathName",
)

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeAlias, overload

from ._yaml import DeferredYaml


class Placeholders:
    MAPPING: ClassVar[Literal[":"]] = ":"
    SEQUENCE: ClassVar[Literal["*"]] = "*"


SchemaPathTerm: TypeAlias = str


class SchemaPath(Sequence[SchemaPathTerm]):
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

    @overload
    def __getitem__(self, i: int) -> SchemaPathTerm: ...

    @overload
    def __getitem__(self, i: slice) -> SchemaPath: ...

    def __getitem__(self, i: int | slice) -> SchemaPathTerm | SchemaPath:
        if type(i) is slice:
            return SchemaPath(*self._terms[i])
        return self._terms[i]

    def __len__(self) -> int:
        return len(self)

    def resolve(self, tree: Any) -> Iterator[tuple[Any, dict[SchemaPath, str | int]]]:
        return self._resolve(0, tree, {})

    def _resolve(
        self,
        depth: int,
        tree: Any,
        substitutions: dict[SchemaPath, str | int],
    ) -> Iterator[tuple[Any, dict[SchemaPath, str | int]]]:
        while depth < len(self._terms):
            if isinstance(tree, DeferredYaml):
                nested = tree.data
            else:
                nested = tree
            match self._terms[depth]:
                case Placeholders.MAPPING:
                    for k, v in nested.items():
                        yield from self._resolve(
                            depth + 1, v, substitutions | {SchemaPath(*self._terms[: depth + 1]): k}
                        )
                case Placeholders.SEQUENCE:
                    for i, v in enumerate(nested):
                        yield from self._resolve(
                            depth + 1, v, substitutions | {SchemaPath(*self._terms[: depth + 1]): i}
                        )
                case str() as key:
                    if key.isdigit():
                        tree = nested[int(key)]
                    else:
                        tree = nested[key]
                case _:
                    raise AssertionError()
            depth += 1
        yield tree, substitutions


@dataclasses.dataclass
class SchemaPathName:
    template: str
    substitutions: tuple[SchemaPath, ...] = ()

    @property
    def is_concrete(self) -> bool:
        return not self.substitutions

    def format(self, subs: Mapping[SchemaPath, str]) -> str:
        return self.template.format(*[subs[k] for k in self.substitutions])
