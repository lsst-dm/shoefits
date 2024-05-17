from __future__ import annotations

__all__ = ("Frame",)

from typing import Any

from ._geom import Box
from ._struct import Struct


class Frame(Struct):
    def __init__(self, bbox: Box, **kwargs: Any):
        if bbox is None:
            raise TypeError("Frame cannot be constructed without a bounding box.")
        super().__init__(bbox, **kwargs)
        self._frame_bbox = bbox

    @property
    def bbox(self) -> Box:
        return self._frame_bbox
