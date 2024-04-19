from __future__ import annotations

__all__ = (
    "FrameInterface",
    "GeneralFrame",
    "DetectorFrame",
    "TractFrame",
    "RootFrame",
    "CutoutFrame",
    "Frame",
)

from typing import Annotated, Literal, Protocol, TypeAlias, Union, cast, TYPE_CHECKING
import pydantic

from ._geom import Box, Extent, Point
from ._skymap import SkyMap
from .schema import (
    FrameSchemaInterface,
    GeneralFrameSchema,
    DetectorFrameSchema,
    TractFrameSchema,
    CutoutFrameSchema,
)


class FrameInterface(FrameSchemaInterface, Protocol):
    @property
    def name(self) -> str: ...

    @property
    def size(self) -> Extent: ...

    @property
    def start(self) -> Point: ...

    @property
    def parent(self) -> FrameInterface | None: ...

    @property
    def bbox(self) -> Box: ...

    @property
    def is_resolved(self) -> Literal[True]: ...

    def cutout(self, box: Box) -> CutoutFrame: ...


class GeneralFrame(GeneralFrameSchema):
    name: str
    start: Point
    size: Extent

    @property
    def bbox(self) -> Box:
        return Box.from_size(self.size, self.start)

    @property
    def is_resolved(self) -> Literal[True]:
        return True

    def cutout(self, box: Box) -> CutoutFrame:
        return CutoutFrame(parent=self, start=box.start, size=box.size)


class DetectorFrame(DetectorFrameSchema):
    instrument: str
    detector: int
    visit: int | None = None
    start: Point
    size: Extent

    @property
    def name(self) -> str:
        return cast(str, super().name)

    @property
    def bbox(self) -> Box:
        return Box.from_size(self.size, self.start)

    @property
    def parent(self) -> None:
        return None

    @property
    def is_resolved(self) -> Literal[True]:
        return True

    def cutout(self, box: Box) -> CutoutFrame:
        return CutoutFrame(parent=self, start=box.start, size=box.size)


# If we add add AmplifierFrame in the future, it could have all of the
# overscan/flip geometry, but it probably shouldn't have a DetectorFrame as its
# parent because they need to be related by a coordinate transform (and even
# then only within a window).


class TractFrame(TractFrameSchema):
    skymap: SkyMap
    tract: int
    patch: int | None = None
    cell: int | None = None

    @property
    def name(self) -> str:
        return cast(str, super().name)

    @property
    def bbox(self) -> Box:
        return cast(Box, super().bbox)

    @property
    def size(self) -> Extent:
        return cast(Extent, super().size)

    @property
    def start(self) -> Point:
        return cast(Point, super().start)

    @property
    def parent(self) -> TractFrame | None:
        if self.cell is not None:
            return self.model_copy(update=dict(cell=None))
        if self.patch is not None:
            return self.model_copy(update=dict(patch=None))
        return None

    @property
    def is_resolved(self) -> Literal[True]:
        return True

    def cutout(self, box: Box) -> CutoutFrame:
        return CutoutFrame(parent=self, start=box.start, size=box.size)


_RootFrame: TypeAlias = Union[GeneralFrame, DetectorFrame, TractFrame]
RootFrame: TypeAlias = Annotated[_RootFrame, pydantic.Field(discriminator="schema_type")]


class CutoutFrame(CutoutFrameSchema):
    parent: RootFrame
    start: Point
    size: Extent

    @property
    def name(self) -> str:
        return cast(str, super().name)

    @property
    def bbox(self) -> Box:
        return Box.from_size(self.size, self.start)

    @property
    def is_resolved(self) -> Literal[True]:
        return True

    def cutout(self, box: Box) -> CutoutFrame:
        return CutoutFrame(parent=self.parent, start=box.start, size=box.size)

    @property
    def slices(self) -> tuple[slice, slice]:
        y0 = self.start.y - self.parent.start.y
        x0 = self.start.x - self.parent.start.x
        return (slice(y0, y0 + self.size.y), slice(x0, x0 + self.size.x))


Frame: TypeAlias = Annotated[Union[_RootFrame, CutoutFrame], pydantic.Field(discriminator="schema_type")]


if TYPE_CHECKING:
    # Check that the types defined in this module actually satisfy the
    # protocols we want; MyPy will complain about these assignments if they do
    # not.
    _f: type[FrameInterface]
    _f = GeneralFrame
    _f = DetectorFrame
    _f = TractFrame
    _f = CutoutFrame
