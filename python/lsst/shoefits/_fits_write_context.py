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
from ._fits_options import FitsCompression, FitsOptions, MaskHeaderStyle
from ._geom import Point
from ._write_context import WriteContext

if TYPE_CHECKING:
    from ._image import Image
    from ._mask import Mask, MaskSchema
    from ._polymorphic import PolymorphicAdapterRegistry

FORMAT_VERSION = (0, 0, 1)


@dataclasses.dataclass
class FitsExtension:
    frame_header: astropy.io.fits.Header | None
    extension_only_header: astropy.io.fits.Header
    array: np.ndarray
    compression: FitsCompression | None = None


class FitsWriteContext(WriteContext):
    def __init__(self, adapter_registry: PolymorphicAdapterRegistry):
        self._primary_header = astropy.io.fits.Header()
        self._primary_header.set(
            keywords.FORMAT_VERSION, ".".join(str(v) for v in FORMAT_VERSION), "SHOEFITS format version."
        )
        self._primary_header["EXTNAME"] = "INDEX"
        self._extensions: list[FitsExtension] = []
        self._adapter_registry = adapter_registry
        self._header_stack: list[astropy.io.fits.Header] = []
        self._fits_options_stack: list[FitsOptions] = []
        self._extlevel: int = 1
        self._extname_counter: Counter[str] = Counter()

    @property
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        return self._adapter_registry

    @contextmanager
    def fits_write_options(self, options: FitsOptions) -> Iterator[None]:
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
        with self._current_header() as current_header:
            current_header.set(f"HIERARCH {key}" if hierarch else key, value, comment)

    def export_header_update(self, header: astropy.io.fits.Header, for_read: bool = False) -> None:
        if for_read and self._header_stack:
            if header:
                warnings.warn(
                    "Header field is nested within a frame other than the root of the tree "
                    "being written to disk, and hence cannot be populated on read. Clear the header field "
                    "writing to avoid this warning (there is no warning on read)."
                )
        with self._current_header() as current_header:
            current_header.update(header)

    def add_image(self, image: Image) -> str | int:
        label = self._get_next_extension_label()
        extension = self._append_extension(label, image.array, start=image.bbox.start, compression=None)
        if image.unit is not None:
            extension.extension_only_header["BUNIT"] = image.unit.to_string(format="fits")
        return f"fits:{label}"

    def add_mask(self, mask: Mask) -> str | int:
        label = self._get_next_extension_label()
        extension = self._append_extension(label, mask.array, start=mask.bbox.start, compression=None)
        self._add_mask_schema_header(extension.extension_only_header, mask.schema)
        return f"fits:{label}"

    def add_array(self, array: np.ndarray) -> str | int:
        label = self._get_next_extension_label()
        self._append_extension(label, array, start=None, compression=None)
        return f"fits:{label}"

    def write(self, root: pydantic.BaseModel, stream: BinaryIO, indent: int | None = 0) -> None:
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
                tile_shape = extension.compression.tile_size.shape + extension.array.shape[2:]
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

    @contextmanager
    def _current_header(self) -> Iterator[astropy.io.fits.Header]:
        if self._header_stack:
            header = self._header_stack[-1]
        else:
            header = self._primary_header
        # Squash astropy warnings about needing HIERARCH, since the only way to
        # get it to do HIERARCH *only when needed* is to let it warn.
        with warnings.catch_warnings(category=astropy.io.fits.verify.VerifyWarning, action="ignore"):
            yield header

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

    def _append_extension(
        self,
        label: keywords.FitsExtensionLabel | int,
        array: np.ndarray,
        start: Point | None = None,
        compression: FitsCompression | None = None,
        parent_header: astropy.io.fits.Header | None = None,
    ) -> FitsExtension:
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
        if start is not None:
            self._add_array_start_wcs(start, extension_only_header)
        extension = FitsExtension(
            array=array,
            frame_header=parent_header,
            extension_only_header=extension_only_header,
            compression=compression,
        )
        self._extensions.append(extension)
        self._primary_header.set(keywords.EXT_ADDRESS.format(ext_index), 0, "Address of extension data.")
        return extension

    def _add_array_start_wcs(self, start: Point, header: astropy.io.fits.Header, wcs_name: str = "A") -> None:
        header.set(f"CTYPE1{wcs_name}", "LINEAR", "Type of projection")
        header.set(f"CTYPE2{wcs_name}", "LINEAR", "Type of projection")
        header.set(f"CRPIX1{wcs_name}", 1.0, "Column Pixel Coordinate of Reference")
        header.set(f"CRPIX2{wcs_name}", 1.0, "Row Pixel Coordinate of Reference")
        header.set(f"CRVAL1{wcs_name}", start.x, "Column pixel of Reference Pixel")
        header.set(f"CRVAL2{wcs_name}", start.y, "Row pixel of Reference Pixel")
        header.set(f"CUNIT1{wcs_name}", "PIXEL", "Column unit")
        header.set(f"CUNIT2{wcs_name}", "PIXEL", "Row unit")

    def _add_mask_schema_header(self, header: astropy.io.fits.Header, schema: MaskSchema) -> None:
        if not self._fits_options_stack:
            options = FitsOptions()
        else:
            options = self._fits_options_stack[-1]
        if options.mask_header_style is MaskHeaderStyle.AFW:
            for mask_plane_index, mask_plane in enumerate(schema):
                if mask_plane is not None:
                    header.set(
                        keywords.AFW_MASK_PLANE.format(mask_plane.name.upper()),
                        mask_plane_index,
                        mask_plane.description,
                    )
