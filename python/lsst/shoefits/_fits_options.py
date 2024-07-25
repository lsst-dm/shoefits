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
    "Fits",
    "FitsCompression",
    "FitsHeaderPropagation",
    "FitsCompressionAlgorithm",
    "MaskHeaderStyle",
    "FitsOptions",
    "FitsOptionsDict",
    "ExportFitsHeaderKey",
)

import dataclasses
from contextlib import ExitStack
from typing import Any, Literal, TypeAlias, TypedDict, Unpack

import astropy.io.fits
import pydantic
import pydantic_core.core_schema as pcs

from ._write_context import WriteContext

FitsCompressionAlgorithm: TypeAlias = Literal["GZIP_1", "GZIP_2"]

FitsHeaderPropagation: TypeAlias = Literal["inherit", "share", "reset"]


@dataclasses.dataclass
class FitsCompression:
    tile_shape: tuple[int, int]
    algorithm: FitsCompressionAlgorithm = "GZIP_2"


MaskHeaderStyle: TypeAlias = Literal["afw"]


class FitsOptionsDict(TypedDict, total=False):
    extname: str | None
    mask_header_style: MaskHeaderStyle | None
    compression: FitsCompression | None
    inherit_primary_header: bool
    parent_header: FitsHeaderPropagation
    parent_wcs: FitsHeaderPropagation
    default_wcs_key: str
    add_wcs: bool | None
    offset_wcs_key: str | None


@dataclasses.dataclass(frozen=True)
class FitsOptions:
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
    description controlled by this option is never read by ShoeFits.
    """

    compression: FitsCompression | None = None
    """How to compress FITS array data."""

    inherit_primary_header: bool = True
    """Whether to add INHERIT='T' to FITS extension headers."""

    parent_header: FitsHeaderPropagation = "share"
    """How to handle FITS header keys exported in the context above this one.

    If "inherit", keys from the context above this one will be added to the
    headers of FITS extensions created in this context, but keys from this
    context will not be propagated up.

    If "share", keys from the context above this one and keys exported in this
    context are combined and appear in all FITS extensions created in either
    context.

    If "reset", keys from the context above this one are ignored.

    The top-level `pydantic.BaseModel` or `Struct` being serialized always
    exports to the primary FITS HDU's header, and this is not considered a
    regular parent (it uses INHERIT='T' instead to share its keys), so this
    option only comes into play two or more levels of `Struct` and/or
    `FitsOptions` nesting below that top-level object.
    """

    parent_wcs: FitsHeaderPropagation = "share"
    """How to handle FITS WCSs exported in the context above this one.

    The options are the same as for `parent_header`, but:
    - A WCS is never added to the primary HDU header or any binary table HDU.
    - Because the primary HDU can never have a WCS, INHERIT='T' is irrelevant,
      and this option comes into play just one level of nesting below the
      top-level object.

    WCS headers from different contexts can only be merged if there are no
    clashes between WCS "keys" (A-Z suffix or ' ').
    """

    default_wcs_key: str = " "
    """Key for exported FITS WCS objects (A-Z or ' ') that do not specify their
    own key.
    """

    add_wcs: bool | None = None
    """Whether to add a FITS WCS to create image HDUs.

    The default (`None`) is to add a FITS WCS for `Image` and `Mask` objects
    but not regular arrays.
    """

    offset_wcs_key: str | None = "A"
    """Key used to create a FITS WCS representing the `Image.start` and
    `Mask.start` integer offsets (A-Z or ' ').

    Set to `None` to disable adding this WCS entirely.
    """


class Fits:
    """Options for reading and writing model attributes to FITS."""

    def __init__(self, **kwargs: Unpack[FitsOptionsDict]):
        self._kwargs = kwargs

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        # This is the Pydantic hook that makes it pay attention to a value in
        # typing.Annotated.
        #
        # In this case we add a "wrap" serializer function that delegates to
        # the default serialization logic for the annotated type after setting
        # the FITS options.
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
        with ExitStack() as stack:
            if write_context := WriteContext.from_info(info):
                if fits_options := write_context.get_fits_options():
                    fits_options = dataclasses.replace(fits_options, **self._kwargs)
                    stack.enter_context(write_context.fits_options(fits_options))
            return handler(obj)


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

    def _serialize(
        self,
        obj: Any,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            header = astropy.io.fits.Header()
            header.set(f"HIERARCH {self.key}" if self.hierarch else self.key, obj, self.comment)
            write_context.export_fits_header(header)
        return handler(obj)
