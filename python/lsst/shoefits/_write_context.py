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

__all__ = ("WriteContext", "WriteError")

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import astropy.io.fits
import numpy as np
import pydantic

if TYPE_CHECKING:
    from ._fits_options import FitsOptions
    from ._image import Image
    from ._mask import Mask
    from ._polymorphic import PolymorphicAdapterRegistry


class WriteError(RuntimeError):
    pass


class WriteContext(ABC):
    @staticmethod
    def from_info(info: pydantic.SerializationInfo) -> WriteContext | None:
        if info.context is None:
            return None
        return info.context.get("shoefits.write_context")

    def inject(self, mapping: dict[str, Any] | None = None) -> dict[str, Any]:
        if mapping is None:
            mapping = {}
        mapping["shoefits.write_context"] = self
        return mapping

    @property
    @abstractmethod
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        raise NotImplementedError()

    @abstractmethod
    def fits_write_options(self, options: FitsOptions) -> AbstractContextManager[None]:
        raise NotImplementedError()

    @abstractmethod
    def export_header_key(
        self,
        key: str,
        value: int | str | float | bool,
        comment: str | None = None,
        hierarch: bool = False,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def export_header_update(self, header: astropy.io.fits.Header, for_read: bool = False) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_image(self, image: Image) -> str | int:
        raise NotImplementedError()

    @abstractmethod
    def add_mask(self, mask: Mask) -> str | int:
        raise NotImplementedError()

    @abstractmethod
    def add_array(self, array: np.ndarray) -> str | int:
        raise NotImplementedError()
