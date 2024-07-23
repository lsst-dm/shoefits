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

__all__ = ("Model",)

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Self

import pydantic

from ._read_context import ReadContext
from ._write_context import WriteContext

if TYPE_CHECKING:
    import astropy.io.fits

    from ._fits_options import FitsOptions


class Model(pydantic.BaseModel):
    """An intermediate base class for ShoeFits models that provides extra
    control over serialization.
    """

    @classmethod
    def _shoefits_nest(cls) -> bool:
        """Return a whether to consider this type a logical level of nesting in
        the serialized form.

        For FITS serialization, this controls whether FITS headers exported by
        this type or fields within it are included only in HDUs generated from
        this type's fields (`True`, default) vs. included also in parent
        or sibling HDUs (`nest=False`).  It also increments the EXTLEVEL
        header value for nested HDUs.
        """
        return True

    def _shoefits_export_fits_header(self) -> astropy.io.fits.Header | None:
        """Return a FITS header to export with this object.

        This will only be called if the output serialization format is actually
        FITS.  Which HDUs will receive this header is controlled by
        `_shoefits_nest`.
        """
        return None

    @classmethod
    def _shoefits_strip_fits_header(cls, header: astropy.io.fits.Header) -> None:
        """Strip any FITS header keys that may have been exported by this
        object.

        This will only be called if the output serialization format is actually
        FITS.
        """
        pass

    def _shoefits_set_fits_write_options(self, options: FitsOptions) -> FitsOptions:
        """Set this FITS write options for all of this model's fields.

        Parameters
        ----------
        options
            The original options, as set by defaults and parent objects.

        Returns
        -------
        options
            The new options to use for this object's fields.
        """
        return options

    @pydantic.model_serializer(mode="wrap")
    def _shoefits_serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> Any:
        with ExitStack() as stack:
            if write_context := WriteContext.from_info(info):
                if self._shoefits_nest():
                    stack.enter_context(write_context.nested())
                if fits_write_options := write_context.get_fits_write_options():
                    stack.enter_context(write_context.fits_write_options(fits_write_options))
                if header := self._shoefits_export_fits_header():
                    write_context.export_fits_header(header)
            return handler(self)

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _shoefits_validate(
        cls, value: Any, handler: pydantic.ValidatorFunctionWrapHandler, info: pydantic.ValidationInfo
    ) -> Self:
        with ExitStack() as stack:
            if read_context := ReadContext.from_info(info):
                if cls._shoefits_nest():
                    stack.enter_context(read_context.nested())
                if header := read_context.primary_header:
                    cls._shoefits_strip_fits_header(header)
            return handler(value)
        raise AssertionError("context stack should not suppress exceptions")
