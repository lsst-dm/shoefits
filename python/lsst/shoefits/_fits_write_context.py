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

__all__ = ()

import dataclasses
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO

import astropy.io.fits
import numpy as np
import pydantic

from . import keywords
from ._dtypes import NumberType
from ._fits_options import FitsCompression, FitsOptions
from ._write_context import WriteContext
from .asdf_utils import ArrayReferenceModel

if TYPE_CHECKING:
    from ._polymorphic import PolymorphicAdapterRegistry

FORMAT_VERSION = (0, 0, 1)


@dataclasses.dataclass
class _FitsFrame:
    header: astropy.io.fits.Header = dataclasses.field(default_factory=astropy.io.fits.Header)
    wcs: astropy.io.fits.Header | None = None


@dataclasses.dataclass
class _FitsExtension:
    frame_stack: tuple[_FitsFrame, ...]
    extension_only_header: astropy.io.fits.Header
    array: np.ndarray
    wcs: bool
    compression: FitsCompression | None = None


class FitsWriteContext(WriteContext):
    """A `WriteContext` implementation for FITS files.

    Parameters
    ----------
    polymorphic_adapter_registry
        Registry of polymorphic adapters that can be used to read fields
        annotated with `Polymorphic`.

    Notes
    -----
    Along with `FitsReadContext`, this defines a FITS-based file format in
    which the primary HDU is a 1-d uint8 array holding a UTF-8 JSON tree.
    Extension HDUs are used to store numeric arrays, including `Image` and
    `Mask` nested within the tree.

    Constructing an instance of this class only sets up some empty in-memory
    data structures.  The `write` method can then be called to invoke
    the Pydantic serialization machinery, which produces the JSON tree,
    gathers arrays and headers to be written to other extension HDUs, and
    writes all of these to FITS.
    """

    def __init__(self, polymorphic_adapter_registry: PolymorphicAdapterRegistry):
        self._primary_header = astropy.io.fits.Header()
        self._primary_header.set(
            keywords.FORMAT_VERSION, ".".join(str(v) for v in FORMAT_VERSION), "SHOEFITS format version."
        )
        self._primary_header["EXTNAME"] = "INDEX"
        self._frame_stack: list[_FitsFrame] = [_FitsFrame(self._primary_header)]
        self._extensions: list[_FitsExtension] = []
        self._adapter_registry = polymorphic_adapter_registry
        self._fits_options_stack: list[FitsOptions] = []
        self._extlevel: int = 1
        self._extname_counter: Counter[str] = Counter()

    @property
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        # Docstring inherited.
        return self._adapter_registry

    def write(self, root: pydantic.BaseModel, stream: BinaryIO, indent: int | None = 0) -> None:
        """Serialize an object to the given stream via FITS.

        Parameters
        ----------
        root
            Object to save.  Must be a `pydantic.BaseModel` instance.
        stream
            Stream to write to, opened for binary output.  Must support seeks.
        indent, optional
            Number of indentation characters to when formatting JSON tree, or
            `None` for minimal whitespace.
        """
        tree_str = root.model_dump_json(indent=indent, context=self.inject())
        asdf_buffer = BytesIO()
        asdf_buffer.write(tree_str.encode())
        tree_size = asdf_buffer.tell()
        asdf_array = np.frombuffer(asdf_buffer.getbuffer(), dtype=np.uint8)
        primary_hdu = astropy.io.fits.PrimaryHDU(asdf_array, header=self._frame_stack[0].header)
        primary_hdu.header.set(keywords.TREE_SIZE, tree_size, "Size of tree in bytes.")
        hdu_list = astropy.io.fits.HDUList([primary_hdu])
        # There's no method to get the size of the header without stringifying
        # it, so that's what we do (here and later).
        address = primary_hdu.filebytes()
        for index, extension in enumerate(self._extensions):
            full_header = astropy.io.fits.Header()
            wcs = None
            for frame in extension.frame_stack:
                full_header.update(frame.header)
                if frame.wcs:
                    wcs = frame.wcs
            if extension.wcs and wcs:
                full_header.update(wcs)
            full_header.set("INHERIT", True)
            full_header.update(extension.extension_only_header)
            if extension.compression:
                tile_shape = extension.compression.tile_shape + extension.array.shape[2:]
                hdu = astropy.io.fits.CompImageHDU(
                    extension.array,
                    header=full_header,
                    compression_type=extension.compression.algorithm.value,
                    tile_shape=tile_shape,
                )
                raise NotImplementedError(
                    "TODO: this triggers some internal error in astropy when we try "
                    "to write the HDUList later.  Not sure what's unusual here."
                )
            else:
                hdu = astropy.io.fits.ImageHDU(extension.array, full_header)
            primary_hdu.header[keywords.EXT_ADDRESS.format(index + 1)] = address + len(hdu.header.tostring())
            address += hdu.filebytes()
            hdu_list.append(hdu)
        hdu_list.writeto(stream)  # TODO: make space for, then add checksums

    def get_fits_write_options(self) -> FitsOptions:
        # Docstring inherited.
        if self._fits_options_stack:
            return self._fits_options_stack[-1]
        else:
            return FitsOptions()

    @contextmanager
    def fits_write_options(self, options: FitsOptions) -> Iterator[None]:
        # Docstring inherited.
        self._fits_options_stack.append(options)
        yield
        del self._fits_options_stack[-1]

    @contextmanager
    def nested(self) -> Iterator[None]:
        self._frame_stack.append(_FitsFrame())
        self._extlevel += 1
        yield
        self._extlevel -= 1
        del self._frame_stack[-1]

    def export_fits_header(
        self, header: astropy.io.fits.Header, for_read: bool = False, is_wcs: bool = False
    ) -> None:
        # Docstring inherited.
        if for_read and len(self._frame_stack) > 1:
            if header:
                warnings.warn(
                    "Header field is nested within a frame other than the root of the tree "
                    "being written to disk, and hence cannot be populated on read. Clear the header field "
                    "before writing to avoid this warning (there is no warning on read)."
                )
        if for_read and is_wcs:
            raise TypeError("for_read and is_wcs cannot both be True.")
        if is_wcs:
            self._frame_stack[-1].wcs = header.copy()
        else:
            self._frame_stack[-1].header.update(header)

    def add_array(
        self,
        array: np.ndarray,
        header: astropy.io.fits.Header | None = None,
        use_wcs_default: bool = False,
    ) -> ArrayReferenceModel:
        # Docstring inherited.
        label = self._get_next_extension_label()
        ext_index = len(self._extensions) + 1
        extension_only_header = astropy.io.fits.Header()
        if isinstance(label, keywords.FitsExtensionLabel):
            label.update_header(extension_only_header)
            self._primary_header.set(
                keywords.EXT_LABEL.format(ext_index),
                str(label),
                "Label for extension used in tree.",
            )
            self._extname_counter[label.extname] += 1
        else:
            self._primary_header.set(
                keywords.EXT_LABEL.format(ext_index),
                label,
                "Label for extension used in tree.",
            )
        extension_only_header["EXTLEVEL"] = self._extlevel
        if header is not None:
            extension_only_header.update(header)
        compression: FitsCompression | None = None
        options = self.get_fits_write_options()
        if options.compression:
            raise NotImplementedError("FITS compression is not yet supported.")
        extension = _FitsExtension(
            array=array,
            frame_stack=tuple(self._frame_stack),
            extension_only_header=extension_only_header,
            compression=compression,
            wcs=(options.wcs if options.wcs is not None else use_wcs_default),
        )
        self._extensions.append(extension)
        self._primary_header.set(keywords.EXT_ADDRESS.format(ext_index), 0, "Address of extension data.")
        return ArrayReferenceModel(
            source=f"fits:{label}", shape=list(array.shape), datatype=NumberType.from_numpy(array.dtype)
        )

    def _get_next_extension_label(self) -> keywords.FitsExtensionLabel | int:
        if self._fits_options_stack:
            options = self._fits_options_stack[-1]
            if options.extname is not None:
                return keywords.FitsExtensionLabel(
                    options.extname,
                    extver=self._extname_counter[options.extname] + 1,
                )
        # Add one for FITS 1-indexed integer convention, add another one since
        # we haven't added this one yet.
        return len(self._extensions) + 2
