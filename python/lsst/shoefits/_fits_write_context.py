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
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Unpack

import astropy.io.fits
import astropy.wcs
import numpy as np
import pydantic

from . import keywords
from ._dtypes import NumberType
from ._fits_options import FitsCompression, FitsOptions, FitsOptionsDict
from ._write_context import WriteContext, WriteError
from .asdf_utils import ArrayModel, ArrayReferenceModel, ArraySerialization

if TYPE_CHECKING:
    from ._polymorphic import PolymorphicAdapterRegistry

FORMAT_VERSION = (0, 0, 1)


@dataclasses.dataclass
class _Frame:
    options: FitsOptions
    header: astropy.io.fits.Header = dataclasses.field(default_factory=astropy.io.fits.Header)
    wcs_map: dict[str, astropy.wcs.WCS] = dataclasses.field(default_factory=dict)


def add_wcs(wcs_map: dict[str, astropy.wcs.WCS], wcs: astropy.wcs.WCS, key: str = "") -> None:
    if wcs_map.setdefault(key, wcs) is not wcs:
        raise WriteError(f"Multiple FITS WCS exports at the same level with key={key!r}.")


class _FrameStack:
    def __init__(
        self,
        bottom: _Frame,
        stack: list[_Frame] | None = None,
    ) -> None:
        self._bottom = _Frame(FitsOptions()) if bottom is None else bottom
        self._stack: list[_Frame] = [] if stack is None else stack

    @property
    def options(self) -> FitsOptions:
        return self._stack[-1].options if self._stack else self._bottom.options

    def top(self) -> _Frame:
        if not self._stack:
            self._stack.append(self._bottom)
        return self._stack[-1]

    @property
    def primary_header(self) -> astropy.io.fits.Header:
        return self._bottom.header

    def push(self, options: FitsOptions) -> None:
        if not self._stack:
            self._bottom.options = options
            self._stack.append(self._bottom)
        else:
            frame = _Frame(options)
            # If we want to share the parent header or WCS map, use the same
            # instance. But we never share the primary header.
            if options.parent_header == "share" and len(self._stack) > 1:
                frame.header = self._stack[-1].header
            if options.parent_wcs == "share":
                frame.wcs_map = self._stack[-1].wcs_map
            self._stack.append(frame)

    def pop(self) -> None:
        del self._stack[-1]

    def make_extension(
        self, array: np.ndarray, add_wcs: bool, header: astropy.io.fits.Header | None = None
    ) -> _FitsExtension:
        result = _FitsExtension(array, add_wcs, self.options, headers=[], wcs_maps=[])
        if header is not None:
            result.headers.append(header)
        if not self._stack:
            return result
        # Add the header on the top of the stack to the extension, and see what
        # the options at the top of the stack say to do about parent headers.
        if len(self._stack) > 1:
            result.headers.append(self._stack[-1].header)
            propagation = self._stack[-1].options.parent_header
            # Loop over frames from second-from-top to second-from-bottom, i.e.
            # don't include the one we just added or the special primary header
            # that is handled via INHERIT=T.
            for frame in reversed(self._stack[1:-1]):
                match propagation:
                    case "inherit":
                        # We'll merge these headers later, when we write the
                        # extensions.  We can't do it now, because we need to
                        # keep the same instances in the stack so sibling
                        # objects can continue to modify them.
                        result.headers.append(frame.header)
                    case "share":
                        # If the headers are shared, they should already be the
                        # same instance.
                        assert frame.header is result.headers[-1]
                    case "reset":
                        # If we're replacing the header fully, we don't care
                        # about any parent ones.
                        break
        # Add the WCS map on the top of the stack to the extension, and see
        # what the options at the top of the stack say to do about parent WCSs.
        result.wcs_maps.append(self._stack[-1].wcs_map)
        propagation = self._stack[-1].options.parent_wcs
        # Loop over frames from second-from-top to bottom, i.e. don't include
        # the one we just added.  Unlike the primary header, the bottom WCS is
        # not special (our primary HDU is never an image and hence never has a
        # WCS for INHERIT=T to work on).
        for frame in reversed(self._stack[:-1]):
            match propagation:
                case "inherit":
                    # We'll merge these WCS maps later, when we write the
                    # extensions.  We can't do it now, because we need to keep
                    # the same instances in the stack so sibling objects can
                    # continue to modify them.
                    result.wcs_maps.append(frame.wcs_map)
                case "share":
                    # If the WCS maps are shared, they should already be the
                    # same instance.
                    assert frame.wcs_map is result.wcs_maps[-1]
                case "reset":
                    # If we're replacing the WCS map fully, we don't care about
                    # any parent ones.
                    break
        return result


@dataclasses.dataclass
class _FitsExtension:
    array: np.ndarray
    add_wcs: bool
    options: FitsOptions
    headers: list[astropy.io.fits.Header]
    wcs_maps: list[dict[str, astropy.wcs.WCS]]
    start: Sequence[int] | None = None
    compression: FitsCompression | None = None

    def make_full_header(self) -> astropy.io.fits.Header:
        full_header = astropy.io.fits.Header()
        if self.options.inherit_primary_header:
            full_header.set("INHERIT", True)
        for header in self.headers:
            full_header.extend(header, unique=True)
        full_wcs_map = {}
        if self.add_wcs:
            for wcs_map in reversed(self.wcs_maps):
                full_wcs_map.update(wcs_map)
        if self.start is not None and self.options.offset_wcs_key is not None:
            offset_wcs_data: dict[str, int | float | str] = {}
            for i, s in enumerate(reversed(self.start)):
                offset_wcs_data[f"CTYPE{i + 1}"] = "LINEAR"
                offset_wcs_data[f"CRPIX{i + 1}"] = 1.0
                offset_wcs_data[f"CRVAL{i + 1}"] = s
                offset_wcs_data[f"CUNIT{i + 1}"] = "pixel"
            add_wcs(full_wcs_map, astropy.wcs.WCS(offset_wcs_data), self.options.offset_wcs_key)
        if full_wcs_map:
            for key in sorted(full_wcs_map.keys()):
                full_header.update(full_wcs_map[key].to_header(key=key))
        return full_header


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

    def __init__(
        self, polymorphic_adapter_registry: PolymorphicAdapterRegistry, **kwargs: Unpack[FitsOptionsDict]
    ):
        options = FitsOptions()
        if kwargs:
            options = dataclasses.replace(options, **kwargs)
        self._frames = _FrameStack(_Frame(options))
        self._frames.primary_header.set(
            keywords.FORMAT_VERSION, ".".join(str(v) for v in FORMAT_VERSION), "SHOEFITS format version."
        )
        self._tree_header = astropy.io.fits.Header()
        self._tree_header["EXTNAME"] = "tree"
        self._extensions: list[_FitsExtension] = []
        self._adapter_registry = polymorphic_adapter_registry
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
        primary_hdu = astropy.io.fits.PrimaryHDU(None, header=self._frames.primary_header)
        primary_hdu.header[keywords.TREE_ADDRESS] = 0  # placeholder
        hdu_list = astropy.io.fits.HDUList([primary_hdu])
        address = primary_hdu.filebytes()
        for index, extension in enumerate(self._extensions):
            header = extension.make_full_header()
            if extension.compression:
                tile_shape = extension.compression.tile_shape + extension.array.shape[2:]
                hdu = astropy.io.fits.CompImageHDU(
                    extension.array,
                    header=header,
                    compression_type=extension.compression.algorithm,
                    tile_shape=tile_shape,
                )
                raise NotImplementedError(
                    "TODO: this triggers some internal error in astropy when we try "
                    "to write the HDUList later.  Not sure what's unusual here."
                )
            else:
                hdu = astropy.io.fits.ImageHDU(extension.array, header)
            self._tree_header[keywords.EXT_ADDRESS.format(index + 1)] = address + len(hdu.header.tostring())
            address += hdu.filebytes()
            hdu_list.append(hdu)
        primary_hdu.header[keywords.TREE_ADDRESS] = address
        json_buffer = BytesIO()
        json_buffer.write(tree_str.encode())
        json_array = np.frombuffer(json_buffer.getbuffer(), dtype=np.uint8)
        json_column = astropy.io.fits.Column("json", format="PB()", array=json_array[np.newaxis, :])
        json_hdu = astropy.io.fits.BinTableHDU.from_columns([json_column], header=self._tree_header)
        hdu_list.append(json_hdu)
        hdu_list.writeto(stream)  # TODO: make space for, then add checksums

    def get_fits_options(self) -> FitsOptions:
        # Docstring inherited.
        return self._frames.options

    @contextmanager
    def fits_options(self, options: FitsOptions) -> Iterator[None]:
        # Docstring inherited.
        self._frames.push(options)
        yield
        self._frames.pop()

    def export_fits_header(self, header: astropy.io.fits.Header) -> None:
        # Docstring inherited.
        frame = self._frames.top()
        frame.header.update(header)

    def export_fits_wcs(self, wcs: astropy.wcs.WCS, key: str | None = None) -> None:
        # Docstring inherited.
        frame = self._frames.top()
        if key is None:
            key = self._frames.options.default_wcs_key
        add_wcs(frame.wcs_map, wcs, key)

    def add_array(
        self,
        array: np.ndarray,
        fits_header: astropy.io.fits.Header | None = None,
        start: Sequence[int] | None = None,
        add_wcs_default: bool = False,
    ) -> ArrayModel:
        # Docstring inherited.
        label = self._get_next_extension_label()
        if label is None:
            return ArraySerialization.to_model(array)
        ext_index = len(self._extensions) + 1
        header = astropy.io.fits.Header()
        label.update_header(header)
        self._tree_header.set(
            keywords.EXT_LABEL.format(ext_index),
            str(label),
            "Label for extension used in tree.",
        )
        self._extname_counter[label.extname] += 1
        header.update(fits_header)
        options = self.get_fits_options()
        if options.compression:
            raise NotImplementedError("FITS compression is not yet supported.")
        extension = self._frames.make_extension(
            array,
            add_wcs=(options.add_wcs if options.add_wcs is not None else add_wcs_default),
            header=header,
        )
        extension.start = start
        self._extensions.append(extension)
        self._tree_header.set(keywords.EXT_ADDRESS.format(ext_index), 0, "Address of extension data.")
        return ArrayReferenceModel(
            source=f"fits:{label}", shape=list(array.shape), datatype=NumberType.from_numpy(array.dtype)
        )

    def _get_next_extension_label(self) -> keywords.FitsExtensionLabel | None:
        options = self._frames.options
        if options.extname is not None:
            return keywords.FitsExtensionLabel(
                options.extname,
                extver=self._extname_counter[options.extname] + 1,
            )
        return None
