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
