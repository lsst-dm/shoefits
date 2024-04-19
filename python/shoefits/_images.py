from __future__ import annotations

__all__ = ("Image",)


from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt


from ._frames import Frame, CutoutFrame
from .schema import Unit, ImagePixelType
from ._geom import Box


_F = TypeVar("_F", bound="Frame")


class Image(Generic[_F]):
    def __init__(self, array: np.ndarray, frame: _F, unit: Unit | None = None):
        assert array.shape == frame.bbox.size.shape
        self._array = array
        self._frame = frame
        self._unit = unit

    @classmethod
    def from_zeros(cls, dtype: npt.DTypeLike, frame: _F, unit: Unit | None = None) -> Image:
        return cls(np.zeros(frame.bbox.size.shape, dtype=dtype), frame, unit)

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        self._array[:, :] = value

    @property
    def frame(self) -> _F:
        return self._frame

    @property
    def unit(self) -> Unit | None:
        return self._unit

    @property
    def bbox(self) -> Box:
        return self._frame.bbox

    def __getitem__(self, box: Box) -> Image[CutoutFrame]:
        frame = self._frame.cutout(box)
        return Image(self._array[frame.slices], frame, self._unit)

    def __setitem__(self, box: Box, value: Image) -> None:
        self[box].array = value.array


class ImageField:
    def __init__(
        self,
        pixel_type: ImagePixelType,
        frame: type[Frame] | None = None,
        unit: Unit | None = None,
        description: str | None = None,
    ):
        self._pixel_type = pixel_type
        self._frame_type = frame
        self._unit = unit
        self._description = description
