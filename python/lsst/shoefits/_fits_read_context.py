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
from typing import TYPE_CHECKING, Any, BinaryIO

import astropy.io.fits
import numpy as np
import pydantic

from . import asdf_utils
from ._geom import Box
from ._polymorphic import PolymorphicAdapterRegistry
from ._read_context import ReadContext

if TYPE_CHECKING:
    from .json_utils import JsonValue


class FitsReadContext(ReadContext):
    """A `ReadContext` implementation for FITS files.

    Parameters
    ----------
    stream
        Stream to read from.  Must be opened for binary input and must support
        seeks.
    parameters, optional
        String-keyed mapping with read options (usually ways to specify
        partial reads, such as subimages).
    polymorphic_adapter_registry
        Registry of polymorphic adapters that can be used to read fields
        annotated with `Polymorphic`.

    Notes
    -----
    Along with `FitsWriteContext`, this defines a FITS-based file format in
    which the primary HDU is a 1-d uint8 array holding a UTF-8 JSON tree.
    Extension HDUs are used to store numeric arrays, including `Image` and
    `Mask` nested within the tree.

    Constructing an instance of this class causes the primary HDU to be read
    from the given stream.  The `read` method can then be called to invoke
    the Pydantic validation machinery on the JSON tree, which will read arrays
    from other extensions as needed.
    """

    def __init__(
        self,
        stream: BinaryIO,
        parameters: Mapping[str, Any] | None = None,
        *,
        polymorphic_adapter_registry: PolymorphicAdapterRegistry,
    ):
        super().__init__(parameters=parameters, polymorphic_adapter_registry=polymorphic_adapter_registry)
        self._stream = stream
        try:
            self._stream.fileno()
        except OSError:
            self._stream_has_fileno = False
        else:
            self._stream_has_fileno = True
        self._fits: astropy.io.fits.HDUList = astropy.io.fits.open(
            self._stream,
            lazy_load_hdus=True,
            memmap=False,
            cache=False,
        )
        tree_hdu: astropy.io.fits.BinTableHDU = self._fits["tree"]
        self._tree: JsonValue = json.loads(tree_hdu.data[0]["json"].tobytes().decode())

    def read(self, model_type: type[pydantic.BaseModel], component: str | None = None) -> Any:
        """Deserialize the stream.

        Parameters
        ----------
        model_type
            The type of the object serialized to the ``stream`` passed at
            construction, and if ``component=None`` the type of the object
            returned.  Must be a subclass of `pydantic.BaseModel`.
        component, optional
            The name of a component of the object to read instead of the full
            thing.

        Returns
        -------
        obj
            The loaded object.
        """
        if component is None:
            tree = self._tree
        else:
            tree = self.seek_component(self._tree, component)
        return model_type.model_validate(tree, context=self.inject())

    def seek_component(self, tree: JsonValue, component: str) -> JsonValue:
        """Access the raw JSON subtree that corresponds to a component.

        This is a hook that allows derived classes to override the
        interpretation of the ``component`` argument to `read`.  The default
        implementation assumes a top-level JSON mapping key with the name of
        each component.
        """
        return tree[component]  # type: ignore[call-overload,index]

    @contextmanager
    def no_parameter_bbox(self) -> Iterator[None]:
        # Docstring inherited.
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
        # Docstring inherited.
        match array_model:
            case asdf_utils.ArrayReferenceModel(source=str(fits_source)) if fits_source.startswith("fits:"):
                fits_source = fits_source.removeprefix("fits:")
                if fits_source.isnumeric():
                    hdu_index = int(fits_source) - 1
                else:
                    name, ver = fits_source.split(",")
                    hdu_index = self._fits.index_of((name, int(ver)))
            case asdf_utils.ArrayReferenceModel(source=int()):
                raise NotImplementedError("ASDF blocks are not supported by this reader.")
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
