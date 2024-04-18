from __future__ import annotations

import pydantic.json

__all__ = (
    "FrameSchemaInterface",
    "GeneralFrameSchema",
    "DetectorFrameSchema",
    "TractFrameSchema",
    "RootFrameSchema",
    "CutoutFrameSchema",
    "FrameSchema",
    "FrameSchemaType",
)

from typing import Annotated, Literal, Protocol, TypeAlias, Union, TYPE_CHECKING
import pydantic

from .._geom import Box, Extent, Point
from .._skymap import SkyMap
from ._simple import SchemaBase, Sentinals, UNKNOWN

FrameSchemaType: TypeAlias = Literal["general_frame", "detector_frame", "tract_frame", "cutout_frame"]


class FrameSchemaInterface(Protocol):
    @property
    def schema_type(self) -> FrameSchemaType: ...

    @property
    def name(self) -> str | UNKNOWN: ...

    @property
    def size(self) -> Extent | UNKNOWN: ...

    @property
    def start(self) -> Point | UNKNOWN: ...

    @property
    def parent(self) -> FrameSchemaInterface | None: ...

    @property
    def is_resolved(self) -> bool: ...


class GeneralFrameSchema(SchemaBase):
    schema_type: Literal["general_frame"] = "general_frame"
    name: str | UNKNOWN = Sentinals.UNKNOWN
    start: Point | UNKNOWN = Sentinals.UNKNOWN
    size: Extent | UNKNOWN = Sentinals.UNKNOWN

    @property
    def parent(self) -> None:
        return None

    @property
    def is_resolved(self) -> bool:
        return (
            self.name is not Sentinals.UNKNOWN
            and self.start is not Sentinals.UNKNOWN
            and self.size is not Sentinals.UNKNOWN
        )


class DetectorFrameSchema(SchemaBase):
    schema_type: Literal["detector_frame"] = "detector_frame"
    instrument: str | UNKNOWN = Sentinals.UNKNOWN
    detector: int | UNKNOWN = Sentinals.UNKNOWN
    visit: int | None | UNKNOWN = Sentinals.UNKNOWN
    amplifier: int | None | UNKNOWN = Sentinals.UNKNOWN
    start: Point | UNKNOWN = Sentinals.UNKNOWN
    size: Extent | UNKNOWN = Sentinals.UNKNOWN

    @property
    def name(self) -> str | UNKNOWN:
        if not self.is_resolved:
            return Sentinals.UNKNOWN
        if self.visit is None:
            name = f"{self.instrument}:{self.detector}"
        else:
            name = f"{self.instrument}/{self.visit}:{self.detector}"
        if self.amplifier is not None:
            return f"{name}/{self.amplifier}"
        else:
            return name

    @property
    def parent(self) -> None:
        return None

    @property
    def is_resolved(self) -> bool:
        return (
            self.instrument is not Sentinals.UNKNOWN
            and self.detector is not Sentinals.UNKNOWN
            and self.visit is not Sentinals.UNKNOWN
            and self.amplifier is not Sentinals.UNKNOWN
        )


class TractFrameSchema(SchemaBase):
    schema_type: Literal["tract_frame"] = "tract_frame"
    skymap: SkyMap | UNKNOWN = Sentinals.UNKNOWN
    tract: int | UNKNOWN = Sentinals.UNKNOWN
    patch: int | None | UNKNOWN = Sentinals.UNKNOWN
    cell: int | None | UNKNOWN = Sentinals.UNKNOWN

    @property
    def name(self) -> str | UNKNOWN:
        if not self.is_resolved:
            return Sentinals.UNKNOWN
        name = f"{self.skymap.name}[{self.tract}]"
        if self.patch is not None:
            name = f"{name}/{self.patch}"
        if self.cell is not None:
            name = f"{name}/{self.cell}"
        return name

    @property
    def bbox(self) -> Box | UNKNOWN:
        # duplicate is_resolved to get type-checkers to narrow these types.
        if not (
            self.skymap is not Sentinals.UNKNOWN
            and self.tract is not Sentinals.UNKNOWN
            and self.patch is not Sentinals.UNKNOWN
            and self.cell is not Sentinals.UNKNOWN
        ):
            return Sentinals.UNKNOWN
        if self.patch is not None:
            if self.cell is not None:
                return self.skymap.get_cell_outer(self.patch, self.cell)
            return self.skymap.get_patch_outer(self.patch)
        return self.skymap.tract_bounds

    @property
    def size(self) -> Extent | UNKNOWN:
        return self.bbox.size

    @property
    def start(self) -> Point | UNKNOWN:
        return self.bbox.start

    @property
    def parent(self) -> TractFrameSchema | None:
        if self.cell is not None:
            return self.model_copy(update=dict(cell=None))
        if self.patch is not None:
            return self.model_copy(update=dict(patch=None))
        return None

    @property
    def is_resolved(self) -> bool:
        return (
            self.skymap is not Sentinals.UNKNOWN
            and self.tract is not Sentinals.UNKNOWN
            and self.patch is not Sentinals.UNKNOWN
            and self.cell is not Sentinals.UNKNOWN
        )

    @pydantic.model_validator(mode="after")
    def _validate(self) -> TractFrameSchema:
        if self.patch is None and self.cell is not None:
            raise ValueError("'cell' must be None if 'patch' is.")
        return self


_RootFrameSchema: TypeAlias = Union[GeneralFrameSchema, DetectorFrameSchema, TractFrameSchema]
RootFrameSchema: TypeAlias = Annotated[_RootFrameSchema, pydantic.Field(discriminator="field_type")]


class CutoutFrameSchema(SchemaBase):
    schema_type: Literal["cutout_frame"] = "cutout_frame"
    parent: RootFrameSchema
    start: Point | UNKNOWN = Sentinals.UNKNOWN
    size: Extent | UNKNOWN = Sentinals.UNKNOWN

    @property
    def name(self) -> str | UNKNOWN:
        if self.start is Sentinals.UNKNOWN or self.size is Sentinals.UNKNOWN:
            return Sentinals.UNKNOWN
        box = Box.from_size(self.size, self.start)
        return f"{self.parent.name}[x={box.x}, y={box.y}]"

    @property
    def is_resolved(self) -> bool:
        return (
            self.parent.is_resolved
            and self.start is not Sentinals.UNKNOWN
            and self.size is not Sentinals.UNKNOWN
        )


_FrameSchema = Union[_RootFrameSchema, CutoutFrameSchema]
FrameSchema: TypeAlias = Annotated[_FrameSchema, pydantic.Field(discriminator="schema_type")]


if TYPE_CHECKING:
    # Check that the types defined in this module actually satisfy the
    # protocols we want; MyPy will complain about these assignments if they do
    # not.
    _s: type[FrameSchemaInterface]
    _s = GeneralFrameSchema
    _s = DetectorFrameSchema
    _s = TractFrameSchema
    _s = CutoutFrameSchema
