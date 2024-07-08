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

"""Constants and utility code for FITS headers written by this package."""

from __future__ import annotations

__all__ = ()

import dataclasses
from typing import Self

import astropy.io.fits

FORMAT_VERSION = "SHOEFITS"
EXT_ADDRESS = "ADR{:05d}"
EXT_LABEL = "LBL{:05d}"
ASDF_BLOCK_SIZE = "BLK{:05d}"
TREE_SIZE = "TREESIZE"
AFW_MASK_PLANE = "HIERARCH MP_{}"


@dataclasses.dataclass
class FitsExtensionLabel:
    """A FITS EXTNAME and EXTVER header key combination."""

    extname: str
    extver: int | None

    def __str__(self) -> str:
        result = self.extname
        if self.extver is not None:
            result = f"{result},{self.extver}"
        return result

    def update_header(self, header: astropy.io.fits.Header) -> None:
        """Modify a header in place with these keys."""
        header["EXTNAME"] = self.extname
        if self.extver is not None:
            header["EXTVER"] = self.extver

    @classmethod
    def parse(cls, s: str) -> Self:
        """Parse the string representation of this type into a new instance."""
        extname, _, raw_extver = s.partition(",")
        return cls(extname=extname, extver=int(raw_extver) if raw_extver else None)
