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

__all__ = ("TestingWriteContext", "TestingReadContext")

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from .. import asdf_utils
from .._fits_options import FitsOptions
from .._geom import Point
from .._image import Image
from .._mask import Mask
from .._polymorphic import PolymorphicAdapterRegistry
from .._read_context import ReadContext
from .._write_context import WriteContext

if TYPE_CHECKING:
    import astropy.io.fits


class TestingWriteContext(WriteContext):
    """A `WriteContext` implementation that just stores the arrays it is
    given, for use in unit tests.
    """

    __test__ = False

    def __init__(self, adapter_registry: PolymorphicAdapterRegistry):
        self._adapter_registry = adapter_registry
        self.images: dict[str | int, Image] = {}
        self.masks: dict[str | int, Mask] = {}
        self.arrays: dict[str | int, np.ndarray] = {}

    @property
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        return self._adapter_registry

    @contextmanager
    def fits_write_options(self, options: FitsOptions) -> Iterator[None]:
        yield

    def export_header_key(
        self,
        key: str,
        value: int | str | float | bool,
        comment: str | None = None,
        hierarch: bool = False,
    ) -> None:
        pass

    def export_header_update(self, header: astropy.io.fits.Header, for_read: bool = False) -> None:
        pass

    def add_image(self, image: Image) -> str | int:
        key = self.add_array(image.array)
        self.images[key] = image
        return key

    def add_mask(self, mask: Mask) -> str | int:
        key = self.add_array(mask.array)
        self.masks[key] = mask
        return key

    def add_array(self, array: np.ndarray) -> str | int:
        key: str | int = len(self.arrays)
        if key % 2:
            key = str(key)
        self.arrays[key] = array
        return key


class TestingReadContext(ReadContext):
    """A `ReadContext` implementation that is backed by mappings of arrays,
    for use in unit tests.
    """

    __test__ = False

    def __init__(
        self,
        adapter_registry: PolymorphicAdapterRegistry,
        arrays: Mapping[str | int, np.ndarray],
    ):
        self._adapter_registry = adapter_registry
        self.arrays = arrays

    @property
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        return self._adapter_registry

    @contextmanager
    def subheader(self) -> Iterator[None]:
        yield

    @contextmanager
    def no_parameter_bbox(self) -> Iterator[None]:
        yield

    def get_array(
        self,
        array_model: asdf_utils.ArrayModel,
        start: Point,
        x_dim: int = -1,
        y_dim: int = -2,
    ) -> np.ndarray:
        match array_model:
            case asdf_utils.InlineArrayModel():
                return np.array(array_model.data, dtype=array_model.datatype.to_numpy())
            case asdf_utils.ArrayReferenceModel():
                return self.arrays[array_model.source]
            case _:
                raise AssertionError()
