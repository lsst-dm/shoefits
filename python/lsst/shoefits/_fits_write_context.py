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
class _FitsExtension:
    frame_header: astropy.io.fits.Header | None
    extension_only_header: astropy.io.fits.Header
    array: np.ndarray
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
        self._extensions: list[_FitsExtension] = []
        self._adapter_registry = polymorphic_adapter_registry
        self._header_stack: list[astropy.io.fits.Header] = []
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
        primary_hdu = astropy.io.fits.PrimaryHDU(asdf_array, header=self._primary_header)
        primary_hdu.header.set(keywords.TREE_SIZE, tree_size, "Size of tree in bytes.")
        hdu_list = astropy.io.fits.HDUList([primary_hdu])
        # There's no method to get the size of the header without stringifying
        # it, so that's what we do (here and later).
        address = primary_hdu.filebytes()
        for index, extension in enumerate(self._extensions):
            if extension.frame_header is None:
                full_header = astropy.io.fits.Header()
            else:
                full_header = extension.frame_header.copy()
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
        if options.subheader:
            if self._header_stack:
                next_header = self._header_stack[-1].copy()
            else:
                next_header = astropy.io.fits.Header()
            self._header_stack.append(next_header)
            n_extensions = len(self._extensions)
            self._extlevel += 1
        yield
        if options.subheader:
            self._extlevel -= 1
            if len(self._extensions) == n_extensions and self._header_stack[-1]:
                warnings.warn("Frame included FITS header exports but no extension data.")
            del self._header_stack[-1]
        del self._fits_options_stack[-1]

    def export_header_key(
        self,
        key: str,
        value: int | str | float | bool,
        comment: str | None = None,
        hierarch: bool = False,
    ) -> None:
        # Docstring inherited.
        self._current_header.set(f"HIERARCH {key}" if hierarch else key, value, comment)

    def export_header_update(self, header: astropy.io.fits.Header, for_read: bool = False) -> None:
        # Docstring inherited.
        if for_read and self._header_stack:
            if header:
                warnings.warn(
                    "Header field is nested within a frame other than the root of the tree "
                    "being written to disk, and hence cannot be populated on read. Clear the header field "
                    "writing to avoid this warning (there is no warning on read)."
                )
        self._current_header.update(header)

    def add_array(
        self, array: np.ndarray, header: astropy.io.fits.Header | None = None
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
        if self._fits_options_stack and (compression := self._fits_options_stack[-1].compression):
            raise NotImplementedError("FITS compression is not yet supported.")
        extension = _FitsExtension(
            array=array,
            frame_header=self._current_header,
            extension_only_header=extension_only_header,
            compression=compression,
        )
        self._extensions.append(extension)
        self._primary_header.set(keywords.EXT_ADDRESS.format(ext_index), 0, "Address of extension data.")
        return ArrayReferenceModel(
            source=f"fits:{label}", shape=list(array.shape), datatype=NumberType.from_numpy(array.dtype)
        )

    @property
    def _current_header(self) -> astropy.io.fits.Header:
        if self._header_stack:
            return self._header_stack[-1]
        else:
            return self._primary_header

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
