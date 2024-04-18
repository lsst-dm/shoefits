from __future__ import annotations

__all__ = ()

from abc import ABC, abstractmethod
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar, Union, final
import pydantic

from ._geom import Box, Extent, Interval


class FrameBase(pydantic.BaseModel, ABC):
    @property
    @abstractmethod
    def bounds(self) -> Box:
        """Bounds of this frame in its own coordinate system."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def frame_type(self) -> str:
        raise NotImplementedError()


_P = TypeVar("_P", bound=FrameBase)


class ChildFrame(FrameBase, Generic[_P]):
    parent: _P

    @property
    def offset(self) -> Extent:
        """Offset between the coordinates of the parent frame and this frame.

        When no flips are involved, this is the offset that must be added to a
        parent coordinate to yield a coordinate in the child frame within the
        `common` area.  When a coordinate is flipped or binned, this relation
        holds only for ``common.min`` (for binning but no flip) or
        ``common.max`` (flip).

        Note that this is typically zero for simple subimage cutouts, which
        should use the same coordinate system as their parent with smaller
        bounds.
        """
        return Extent(x=0, y=0)

    @property
    def common(self) -> Box:
        """Bounds of the area actually shared by this frame and its parent,
        in the child frame's coordinates.

        This must be contained by `bounds`, but `bounds` may extend beyond
        `common` to represent cases where the parent frame's image is built by
        first cropping a child image (or many child images) and then stitching
        them together, such as raw amplifier images with overscan regions or
        padded coadd cells.
        """
        return self.bounds

    @property
    def flip_x(self) -> bool:
        """If `True`, the pixels within the `common` box must be reversed in
        the x direction to relate them to the parent pixels of the same area.
        """
        return False

    @property
    def flip_y(self) -> bool:
        """If `True`, the pixels within the `common` box must be reversed in
        the y direction to relate them to the parent pixels of the same area.
        """
        return False


@final
class GeneralFrame(FrameBase):
    name: str
    bounds: Box

    frame_type: Literal["general"] = "general"


@final
class CutoutFrame(ChildFrame[_P]):
    bounds: Box

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def frame_type(self) -> str:
        return f"{self.parent.frame_type}_cutout"


@final
class DetectorFrame(FrameBase):
    instrument: str
    detector: int
    visit: int | None
    bounds: Box

    frame_type: Literal["detector"] = "detector"


@final
class AmplifierFrame(ChildFrame[DetectorFrame]):
    amplifier: int

    frame_type: Literal["amplifier"] = "amplifier"

    assembled_data_section: Box
    """Bounding box of the data section of the amplifier in assembled detector
    coordinates.
    """

    @property
    def data_section(self) -> Box:
        """Bounding box of the data section of the amplifier in its own frame.

        By convention, the minimum coordinate of the data section is always
        ``(0, 0)`` in the amplifier's own coordinate system, and this is
        always the first real pixel read out.
        """
        return self.assembled_data_section - self.assembled_data_section.min.as_extent()

    flip_x: bool
    """Whether the amplifier is flipped in the x direction."""

    flip_y: bool
    """Whether the amplifier is flipped in the x direction."""

    horizontal_overscan: Interval
    """Bounds of the horizontal overscan region (in x)."""

    horizontal_prescan: Interval
    """Bounds of the horizontal prescan region (in x)."""

    vertical_overscan: Interval
    """Bounds of the horizontal prescan region (in y)."""

    @property
    def bounds(self) -> Box:
        return Box(
            x=Interval.hull(
                self.horizontal_overscan,
                self.horizontal_prescan,
                self.data_section.x,
            ),
            y=Interval.hull(self.vertical_overscan, self.data_section.y),
        )

    @property
    def offset(self) -> Extent:
        return self.assembled_data_section.min.as_extent()

    @property
    def common(self) -> Box:
        return self.data_section


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
    builder: Literal["legacy"]
    cell_inner_dimensions: Extent
    cell_border: int
    n_cells_per_patch_inner: int
    n_cells_per_patch_border: int


SkyMap: TypeAlias = Annotated[Union[LegacySkyMap, CellSkyMap], pydantic.Field(discriminator="builder")]


@final
class TractFrame(FrameBase):
    skymap: SkyMap
    tract: int

    frame_type: Literal["tract"] = "tract"

    @property
    def bounds(self) -> Box:
        return self.skymap.tract_bounds


class PatchFrame(ChildFrame[TractFrame]):
    patch: int

    frame_type: Literal["patch"] = "patch"

    @property
    def skymap(self) -> SkyMap:
        return self.parent.skymap

    @property
    def tract(self) -> int:
        return self.parent.tract

    @property
    def bounds(self) -> Box:
        return self.parent.skymap.get_patch_outer(self.patch)

    @property
    def common(self) -> Box:
        return self.parent.skymap.get_patch_inner(self.patch)


class CellFrame(ChildFrame[PatchFrame]):
    cell: int

    frame_type: Literal["cell"] = "cell"

    @property
    def skymap(self) -> SkyMap:
        return self.parent.skymap

    @property
    def tract(self) -> int:
        return self.parent.tract

    @property
    def patch(self) -> int:
        return self.parent.patch

    @property
    def bounds(self) -> Box:
        return self.parent.skymap.get_cell_outer(self.patch, self.cell)

    @property
    def common(self) -> Box:
        return self.parent.skymap.get_cell_inner(self.patch, self.cell)


Frame: TypeAlias = Annotated[
    Union[
        GeneralFrame,
        DetectorFrame,
        AmplifierFrame,
        TractFrame,
        PatchFrame,
        CellFrame,
        CutoutFrame[GeneralFrame],
        CutoutFrame[DetectorFrame],
        CutoutFrame[AmplifierFrame],
        CutoutFrame[TractFrame],
        CutoutFrame[PatchFrame],
        CutoutFrame[CellFrame],
    ],
    pydantic.Field(discriminator="frame_type"),
]
