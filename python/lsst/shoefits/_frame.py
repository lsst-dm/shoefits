# This file is part of lsst-shoefits.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

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
