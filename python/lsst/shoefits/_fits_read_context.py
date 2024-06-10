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

__all__ = ("FitsReadContext",)

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar

import astropy.io.fits
import numpy as np
import pydantic

from . import asdf_utils, keywords
from ._dtypes import str_to_numpy
from ._geom import Box, Extent, Point
from ._polymorphic import PolymorphicAdapterRegistry
from ._read_context import ReadContext, ReadError

if TYPE_CHECKING:
    from .json_utils import JsonValue


_M = TypeVar("_M", bound=pydantic.BaseModel)


class FitsReadContext(ReadContext):
    def __init__(
        self,
        buffer: BinaryIO,
        parameters: Mapping[str, Any] | None = None,
        *,
        adapter_registry: PolymorphicAdapterRegistry,
    ):
        self._buffer = buffer
        self._parameters = parameters or {}
        self._ext_addresses: dict[str | int, int] = {}
        self._header = astropy.io.fits.Header()
        fits: astropy.io.fits.Header = astropy.io.fits.open(self._buffer)
        self._read_json_and_addresses(fits[0])
        self._adapter_registry = adapter_registry
        self._frame_depth = 0

    def _read_json_and_addresses(self, hdu: astropy.io.fits.PrimaryHDU) -> None:
        tree_size = hdu.header.pop(keywords.TREE_SIZE)
        self._tree: JsonValue = json.loads(hdu.section[:tree_size].tobytes().decode())
        n = 1
        while (ext_addr := hdu.header.pop(keywords.EXT_ADDRESS.format(n), None)) is not None:
            ext_label = hdu.header.pop(keywords.EXT_LABEL.format(n))
            self._ext_addresses[ext_label] = ext_addr
            n += 1
        self._header.update(hdu.header)
        self._header.strip()

    def read(self, model_type: type[_M], component: str | None = None) -> Any:
        if component is None:
            tree = self._tree
        else:
            tree = self.seek_component(tree, component)
        return model_type.model_validate(tree, context=self.inject())

    def seek_component(self, tree: JsonValue, component: str) -> JsonValue:
        return tree[component]  # type: ignore[call-overload,index]

    def get_parameter_bbox(self, full_bbox: Box, parameters: Mapping[str, Any]) -> Box:
        return parameters.get("bbox", full_bbox)

    @contextmanager
    def subheader(self) -> Iterator[None]:
        self._frame_depth += 1
        yield
        self._frame_depth -= 1

    def get_array(self, array_model: asdf_utils.ArrayModel, start: Point) -> np.ndarray:
        match array_model:
            case asdf_utils.ArrayReferenceModel(source=str(fits_source)):
                address = self._ext_addresses[fits_source.removeprefix("fits:")]
            case asdf_utils.ArrayReferenceModel(source=int()):
                raise NotImplementedError("ASDF block reads not yet supported.")
            case asdf_utils.InlineArrayModel(data=data, datatype=type_name):
                # Inline arrays take a different code path because we have to
                # convert the entire thing to an array and then (maybe) slice,
                # whereas in other cases we do partial reads when slicing.
                array: np.ndarray = np.array(data, dtype=str_to_numpy(type_name))
                full_bbox = Box.from_size(Extent.from_shape(array.shape[-2:]), start=start)
                bbox = self.get_parameter_bbox(full_bbox, self._parameters)
                if bbox != full_bbox:
                    if not full_bbox.contains(bbox):
                        raise ReadError(f"Array has bbox={full_bbox}, which does not contain {bbox}.")
                    array = array[..., bbox.y.slice_within(full_bbox.y), bbox.x.slice_within(full_bbox.x)]
                return array
            case _:
                raise AssertionError()
        full_bbox = Box.from_size(Extent.from_shape(array_model.shape[-2:]), start=start)
        bbox = self.get_parameter_bbox(full_bbox, self._parameters)
        dtype = np.dtype(array_model.datatype).newbyteorder("B" if array_model.byteorder == "big" else "L")
        if not full_bbox.contains(bbox):
            raise ReadError(f"Array has bbox={full_bbox}, which does not contain {bbox}.")
        start_address = address + (bbox.y.start - full_bbox.y.start) * bbox.x.size * dtype.itemsize
        if bbox.x == full_bbox.x:
            # We can read full rows because they're contiguous on disk.
            self._buffer.seek(start_address)
            array1d = np.fromfile(self._buffer, dtype=dtype, offset=0, count=math.prod(array_model.shape))
            array = array1d.reshape(*array_model.shape)
        else:
            # Read row-by-row.  We don't do any clever caching or buffering
            # because we're *hoping* that's best left to the file-like object
            # we're passed.
            array = np.zeros(array_model.shape, dtype=dtype)
            start_address += (bbox.x.start - full_bbox.x.start) * dtype.itemsize
            self._buffer.seek(start_address)
            stride = full_bbox.x.size * dtype.itemsize
            for i in range(bbox.y.size):
                array[i, :] = np.fromfile(self._buffer, dtype=dtype, offset=i * stride, count=bbox.x.size)
        if not dtype.isnative:
            array.byteswap(inplace=True)
        return array
