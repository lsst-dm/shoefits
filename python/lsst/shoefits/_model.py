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

__all__ = ("Struct",)

import dataclasses
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, ClassVar

import astropy.wcs
import pydantic

from ._fits_options import FitsOptions, FitsOptionsDict
from ._write_context import WriteContext

if TYPE_CHECKING:
    import astropy.io.fits


class Struct(pydantic.BaseModel):
    """An intermediate base class for ShoeFits models that provides extra
    control over serialization.
    """

    struct_fits_options: ClassVar[FitsOptionsDict] = FitsOptionsDict(
        parent_header="inherit", parent_wcs="inherit"
    )

    def struct_export_fits_header(self) -> astropy.io.fits.Header | None:
        """Return a FITS header to export with this object.

        This will only be called if the output serialization format is actually
        FITS.
        """
        return None

    def struct_export_fits_wcs(self) -> astropy.wcs.WCS | tuple[astropy.wcs.WCS, str] | None:
        """Return a FITS wCS to export with any images or masks nested under
        this model.

        May return a single `astropy.wcs.WCS` instance or a pair of
        ``(astropy.wcs.WCS, str)``, with the latter single-uppercase letter
        to use as the name of the WCS in the header.

        This will only be called if the output serialization format is actually
        FITS, and only for HDUs marked as accepting a WCS (see
        `FitsOptions.wcs`).
        """
        return None

    def struct_set_fits_options(self, options: FitsOptions) -> FitsOptions:
        """Set this FITS write options for all of this model's fields.

        Parameters
        ----------
        options
            The original options, as set by defaults and parent objects.

        Returns
        -------
        options
            The new options to use for this object's fields.

        Notes
        -----
        The default implementation applies `struct_fits_options`, and should
        generally be called via `super` before additional edits are made.
        """
        if self.struct_fits_options:
            options = dataclasses.replace(options, **self.struct_fits_options)
        return options

    @pydantic.model_serializer(mode="wrap")
    def struct_serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> Any:
        with ExitStack() as stack:
            if write_context := WriteContext.from_info(info):
                if fits_options := write_context.get_fits_options():
                    # If this is not a FITS writer, fits_options is None and
                    # this block is skipped.
                    fits_options = self.struct_set_fits_options(fits_options)
                    stack.enter_context(write_context.fits_options(fits_options))
                    if header := self.struct_export_fits_header():
                        write_context.export_fits_header(header)
                    match self.struct_export_fits_wcs():
                        case [astropy.wcs.WCS() as wcs, key]:
                            write_context.export_fits_wcs(wcs, key)
                        case astropy.wcs.WCS() as wcs:
                            write_context.export_fits_wcs(wcs)
                        case None:
                            pass
                        case _:
                            raise TypeError("Incorrect return type for 'struct_export_fits_wcs'.")
            return handler(self)
