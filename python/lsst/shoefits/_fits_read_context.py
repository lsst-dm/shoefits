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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar

import astropy.io.fits
import numpy as np
import pydantic

from . import asdf_utils, keywords
from ._geom import Box
from ._polymorphic import PolymorphicAdapterRegistry
from ._read_context import ReadContext

if TYPE_CHECKING:
    from .json_utils import JsonValue


_M = TypeVar("_M", bound=pydantic.BaseModel)


class FitsReadContext(ReadContext):
    def __init__(
        self,
        stream: BinaryIO,
        parameters: Mapping[str, Any] | None = None,
        *,
        adapter_registry: PolymorphicAdapterRegistry,
    ):
        super().__init__(parameters=parameters, polymorphic_adapter_registry=adapter_registry)
        self._stream = stream
        try:
            self._stream.fileno()
        except OSError:
            self._stream_has_fileno = False
        else:
            self._stream_has_fileno = True
        self._header = astropy.io.fits.Header()
        self._fits: astropy.io.fits.HDUList = astropy.io.fits.open(
            self._stream,
            lazy_load_hdus=True,
            memmap=False,
            cache=False,
        )
        self._read_json_and_addresses(self._fits[0])
        self._subheader_depth = 0

    def _read_json_and_addresses(self, hdu: astropy.io.fits.PrimaryHDU) -> None:
        # The tree size header is some future-proofing for the possibility of
        # writing ASDF data blacks after the tree as part of the same HDU.  At
        # present, it's identical to NAXIS1.
        tree_size = hdu.header.pop(keywords.TREE_SIZE)
        self._tree: JsonValue = json.loads(hdu.section[:tree_size].tobytes().decode())
        # Strip out extension array-data addresses, since we don't use them at
        # present.  If we reimplement the low-level reading ourselves using
        # these (instead of delegating to astropy) we should be able to avoid
        # doing O(N) seeks and header reads to find an array referenced by the
        # tree.
        n = 1
        while hdu.header.pop(keywords.EXT_ADDRESS.format(n), None) is not None:
            del hdu.header[keywords.EXT_LABEL.format(n)]
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

    @contextmanager
    def subheader(self) -> Iterator[None]:
        self._subheader_depth += 1
        try:
            yield
        finally:
            self._subheader_depth -= 1

    @contextmanager
    def no_parameter_bbox(self) -> Iterator[None]:
        self._no_parameter_bbox_depth += 1
        try:
            yield
        finally:
            self._no_parameter_bbox_depth -= 1

    def get_array(
        self,
        array_model: asdf_utils.ArrayModel,
        bbox_from_shape: Callable[[tuple[int, ...]], Box] = Box.from_shape,
        slice_result: Callable[[Box], tuple[slice, ...]] | None = None,
    ) -> np.ndarray:
        match array_model:
            case asdf_utils.ArrayReferenceModel(source=str(fits_source)) if fits_source.startswith("fits:"):
                fits_source = fits_source.removeprefix("fits:")
                if fits_source.isnumeric():
                    hdu_index = int(fits_source) - 1
                else:
                    name, ver = fits_source.split(",")
                    hdu_index = self._fits.index_of((name, int(ver)))
            case asdf_utils.ArrayReferenceModel(source=int()):
                raise NotImplementedError("ASDF blocks not supported by this reader.")
            case asdf_utils.InlineArrayModel(data=data, datatype=type_enum):
                # Inline arrays take a different code path because we have to
                # convert the entire thing to an array and then (maybe) slice,
                # whereas in other cases we do partial reads when slicing.
                array: np.ndarray = np.array(data, dtype=type_enum.to_numpy())
                full_bbox = bbox_from_shape(array.shape)
                if slice_result is None:
                    return array
                else:
                    return array[slice_result(full_bbox)]
            case _:
                raise AssertionError()
        full_bbox = bbox_from_shape(tuple(array_model.shape))
        if slice_result is None:
            array = self._fits[hdu_index].data
        else:
            array = self._fits[hdu_index].section(slice_result(full_bbox))
        # Logic below tries to avoid unnecessary copies, but to avoid them
        # entirely, we'd have to move this to a compiled language.
        if not array.flags.aligned or not array.flags.writeable:
            array = array.copy()
        if not array.dtype.isnative:
            array = array.newbyteorder().byteswap(inplace=True)
        return array

    def _read_array1d_from_stream(self, dtype: np.dtype, address: int, count: int) -> np.ndarray:
        self._stream.seek(address)
        if self._stream_has_fileno:
            return np.fromfile(self._stream, dtype=dtype, count=count)
        else:
            buffer = self._stream.read(dtype.itemsize * count)
            return np.frombuffer(buffer, dtype=dtype, count=count)
