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

__all__ = (
    "FitsCompression",
    "FitsCompressionAlgorithm",
    "MaskHeaderStyle",
    "FitsOptions",
    "ExportFitsHeaderKey",
)

import dataclasses
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.io.fits
import pydantic
import pydantic_core.core_schema as pcs

from . import keywords
from ._read_context import ReadContext
from ._write_context import WriteContext

if TYPE_CHECKING:
    from ._mask import MaskSchema


class FitsCompressionAlgorithm(enum.StrEnum):
    """Enumeration of FITS compression algorithms."""

    # Enum values are overridden to match the FITS tile compression extension.
    GZIP_1 = "GZIP_1"
    GZIP_2 = "GZIP_2"


@dataclasses.dataclass
class FitsCompression:
    tile_shape: tuple[int, int]
    algorithm: FitsCompressionAlgorithm = FitsCompressionAlgorithm.GZIP_2


class MaskHeaderStyle(enum.StrEnum):
    """Enumeration of ways to represent a `MaskSchema` in a FITS header."""

    AFW = enum.auto()
    """The style used by the `lsst.afw` package.

    This style writes one HIERARCH key for each mask plane, with the key
    the uppercased name of the mask plane with a ``MP_`` prefix, and the
    index of that plane in schema, starting from zero.

    Note that while `lsst.afw` masks have the maximum number of mask planes
    set by the mask pixel type (e.g. an ``int32`` mask has room for 32 planes),
    and hence the index corresponds directly to the bit that is set in the
    image, ShoeFits masks have an extra dimension whose shape multiplies the
    number of planes, and hence converting this the plane index to the bit in
    a particular element of that extra dimension is more complicated (and is
    best left to `MaskSchema.bitmask`, when possible).
    """


@dataclasses.dataclass
class FitsOptions:
    """Options for reading and writing model attributes to FITS.

    Notes
    -----
    This class is intended to be used via `typing.Annotated` to provide
    extra information about how to save an attribute when writing to FITS.
    For example, to set the EXTNAME header for a FITS extension that represents
    an `Image` object::

        from typing import Annotated

        class Struct(pydantic.BaseModel):
            variance: Annotated[Image, FitsOptions(extname="VARIANCE")]

    FITS options may be used to annotate any kind of field, though they may
    have no effect on some (e.g. there is no EXTNAME for simple scalar-value
    field).
    When applied to a model or container field, options will apply to all
    nested members unless they are overridden by a new `FitsOptions`
    annotation at a lower level.
    """

    extname: str | None = None
    """The ``EXTNAME`` header key to use for the FITS extension that holds this
    object's data.

    When multiple fields end up with the same ``EXTNAME`` in a single FITS
    file, they are automatically assigned different ``EXTVER`` values, counting
    up from the first encountered in depth-first order.  This guarantees
    uniqueness of the ``(EXTNAME, EXTVER)`` tuple.
    """

    mask_header_style: MaskHeaderStyle | None = None
    """How to export a description of a `Mask` FITS extension's schema to the
    FITS header.

    The `MaskSchema` description is always saved to the JSON tree in the
    primary FITS HDU, and that description is considered canonical.  The header
    description controlled by this option is never read by SHOEFITS.
    """

    compression: FitsCompression | None = None
    """How to compress FITS array data."""

    wcs: bool | None = None
    """How to associate fields that export a FITS WCS with fields that export
    arrays.

    If `True`, all arrays that are written to image HDUs will have their
    headers populated with FITS WCS headers exported by siblings or parents
    (as defined by `Model._shoefits_nest`).  If `False`, FITS WCS headers are
    never written.  If `None` (default), FITS WCS headers are included for
    `Image` and `Mask` objects, but not arbitrary `numpy.ndarray` fields.
    """

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        # This is the Pydantic hook that makes it pay attention to a value in
        # typing.Annotated.
        #
        # In this case we a serializer function that
        # delegates to the d serialization logic for the
        # annotated type within a context manager provided by the `ReadContext`
        # or `WriteContext` method.
        base_schema = handler(source_type)
        return pcs.json_or_python_schema(
            json_schema=base_schema,
            python_schema=base_schema,
            serialization=pcs.wrap_serializer_function_ser_schema(self._serialize, info_arg=True),
        )

    def _serialize(
        self,
        obj: Any,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            with write_context.fits_write_options(self):
                return handler(obj)
        else:
            return handler(obj)

    def add_array_start_wcs(
        self, header: astropy.io.fits.Header, start: Sequence[int], wcs_name: str = "A"
    ) -> None:
        """Modify a FITS header to include a WCS that applies an integer
        offset.

        Parameters
        ----------
        header
            Header to modify in place.
        start
            Tuple of offset values.  These are the same order as the
            corresponding array's dimensions, which is the reverse of the
            dimensions according to FITS (since FITS uses column-major storage
            while numpy defaults to row-major storage).
        wcs_name, optional
            Single-character FITS WCS name suffix.
        """
        for i, s in enumerate(reversed(start)):
            header.set(f"CTYPE{i + 1}{wcs_name}", "LINEAR")
            header.set(f"CRPIX{i + 1}{wcs_name}", 1.0)
            header.set(f"CRVAL{i + 1}{wcs_name}", s)
            header.set(f"CUNIT{i + 1}{wcs_name}", "PIXEL")

    def add_mask_schema_header(self, header: astropy.io.fits.Header, schema: MaskSchema) -> None:
        """Modify a FITS header to include a description of a `MaskSchema`."""
        if self.mask_header_style is MaskHeaderStyle.AFW:
            for mask_plane_index, mask_plane in enumerate(schema):
                if mask_plane is not None:
                    header.set(
                        keywords.AFW_MASK_PLANE.format(mask_plane.name.upper()),
                        mask_plane_index,
                        mask_plane.description,
                    )


@dataclasses.dataclass
class ExportFitsHeaderKey:
    """An annotation for simple value fields that exports them to FITS headers.

    Notes
    -----
    This class is intended to be used via `typing.Annotated` to provide
    extra information about how to save an attribute when writing to FITS.
    For example, to add a ``VAL`` header keyword to the FITS header for an
    image::

        from typing import Annotated

        class Struct(pydantic.BaseModel):
            image: Image
            value: Annotated[int, ExportFitsHeaderKey("VAL")]

    This annotation should only be used with fields whose types are natively
    compatible with FITS headers (ASCII `str`, `int`, `float`, `bool`).
    """

    key: str
    """FITS header key.

    To avoid warnings, this should be all uppercase and less than 7 characters
    (all ASCII) unless ``hierarch=True``.
    """

    hierarch: bool = False
    """Whether to use the FITS HIERARCH convention for long and case-sensitive
    keywords.
    """

    comment: str = ""
    """Description to include in the FITS header.  Will be truncated (with a
    warning) if too long to fit with the key and value in the 80 char limit.
    """

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        # This is the Pydantic hook that makes it pay attention to a value in
        # typing.Annotated.
        #
        # In this case we add "wrap" a serializer function that delegate sto
        # the default serialization logic for the annotated type after asking
        # the `WriteContext` to export the FITS header key.
        base_schema = handler(source_type)
        return pcs.json_or_python_schema(
            json_schema=base_schema,
            python_schema=base_schema,
            serialization=pcs.wrap_serializer_function_ser_schema(self._serialize, info_arg=True),
        )

    def _validate(
        self, value: Any, handler: pydantic.ValidatorFunctionWrapHandler, info: pydantic.ValidationInfo
    ) -> Any:
        if read_context := ReadContext.from_info(info):
            if header := read_context.primary_header:
                if self.key in header:
                    del header[self.key]

    # TODO: we should have a _validate override too, to strip header keywords
    # from the primary HDU so we can read what's left back in.

    def _serialize(
        self,
        obj: Any,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            header = astropy.io.fits.Header()
            header.set(f"HIERARCH {self.key}" if self.hierarch else self.key, obj, self.comment)
            write_context.export_fits_header(header, for_read=False)
        return handler(obj)
