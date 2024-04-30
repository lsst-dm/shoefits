from __future__ import annotations

import pydantic.json

__all__ = ("SkyMap", "SkyMapDefinition")

from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias, Union

import pydantic

from ._geom import Box, Extent


class SkyMapBase(pydantic.BaseModel, ABC):
    name: str

    @property
    @abstractmethod
    def tract_bounds(self) -> Box:
        raise NotImplementedError()

    @abstractmethod
    def get_patch_outer(self, patch: int) -> Box:
        raise NotImplementedError()

    @abstractmethod
    def get_patch_inner(self, patch: int) -> Box:
        raise NotImplementedError()

    @abstractmethod
    def get_cell_outer(self, patch: int, cell: int) -> Box:
        raise NotImplementedError()

    @abstractmethod
    def get_cell_inner(self, patch: int, cell: int) -> Box:
        raise NotImplementedError()


class LegacySkyMap(SkyMapBase):
    builder: Literal["legacy"]
    patch_inner_dimensions: Extent
    patch_border: int


class CellSkyMap(SkyMapBase):
    builder: Literal["cell"]
    cell_inner_dimensions: Extent
    cell_border: int
    n_cells_per_patch_inner: int
    n_cells_per_patch_border: int


SkyMapDefinition: TypeAlias = Annotated[
    Union[LegacySkyMap, CellSkyMap], pydantic.Field(discriminator="builder")
]

# TODO: make this an object that serializes a JSON pointer and reads from
# validation context so we only store it once.
SkyMap: TypeAlias = SkyMapDefinition
