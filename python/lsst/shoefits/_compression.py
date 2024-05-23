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

__all__ = ("FitsCompression", "FitsCompressionAlgorithm")


import enum

import pydantic

from ._geom import Extent


class FitsCompressionAlgorithm(enum.Enum):
    GZIP_1 = "GZIP_1"
    GZIP_2 = "GZIP_2"


class FitsCompression(pydantic.BaseModel):
    algorithm: FitsCompressionAlgorithm = FitsCompressionAlgorithm.GZIP_2
    tile_size: Extent
