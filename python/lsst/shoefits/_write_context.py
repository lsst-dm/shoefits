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

__all__ = ("WriteContext", "WriteError")

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import astropy.io.fits
import astropy.wcs
import numpy as np
import pydantic

if TYPE_CHECKING:
    from ._fits_options import FitsOptions
    from ._polymorphic import PolymorphicAdapterRegistry
    from .asdf_utils import ArrayModel


class WriteError(RuntimeError):
    """Exception raised when serializing a data structure fails."""


class WriteContext(ABC):
    """Base class for objects that serialize data structures.

    Notes
    -----
    `WriteContext` is designed to be used by invoking Pydantic serialization
    machinery with a `WriteContext` instance included in Pydantic's
    "Serialization Context" dictionary.  Subclasses are expected to have their
    own entry points that depend on how the serialized form is represented in
    Python.
    """

    @staticmethod
    def from_info(info: pydantic.SerializationInfo) -> WriteContext | None:
        """Extract a `WriteContext` instance inside a custom serialization
        function.

        Parameters
        ----------
        info
            The ``info`` argument passed to custom Pydantic serializers.

        Returns
        -------
        read_context
            Extracted write context, or `None` if there was no serialization
            context or no write context within it.
        """
        if info.context is None:
            return None
        return info.context.get("shoefits.write_context")

    def inject(self, mapping: dict[str, Any] | None = None) -> dict[str, Any]:
        """Inject the write context into a dictionary that will be passed as
        the Pydantic serialization context.

        Parameters
        ----------
        mapping, optional
            If provided, an existing dictionary to modify in place and return.

        Returns
        -------
        mapping
            A mapping suitable for use as the Pydantic serialization context.
        """
        if mapping is None:
            mapping = {}
        mapping["shoefits.write_context"] = self
        return mapping

    @property
    @abstractmethod
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        """A registry used to save objects in fields annotated as
        `Polymorphic`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_fits_options(self) -> FitsOptions | None:
        """Return the current `FitsOptions`.

        Non-FITS implementations should return `None`.
        """
        raise NotImplementedError()

    @abstractmethod
    def fits_options(self, options: FitsOptions) -> AbstractContextManager[None]:
        """Return a context manager that applies FITS-specific write options
        to this object any any nested under it.
        """
        raise NotImplementedError()

    def export_fits_header(self, header: astropy.io.fits.Header) -> None:
        """Export FITS header entries.

        Parameters
        ----------
        header
            Header entries to add.
        """
        pass

    def export_fits_wcs(self, wcs: astropy.wcs.WCS, key: str | None = None) -> None:
        """Export FITS header entries.

        Parameters
        ----------
        wcs
            FITS WCS to add.
        key, optional
            Key (a single uppercase letter or the empty string) to distinguish
            this WCS from others in the same header.
        """
        pass

    @abstractmethod
    def add_array(
        self,
        array: np.ndarray,
        fits_header: astropy.io.fits.Header | None = None,
        start: Sequence[int] | None = None,
        add_wcs_default: bool = False,
    ) -> ArrayModel:
        """Write an array.

        Parameters
        ----------
        array
            Array to save.
        fits_header, optional
            Header entries to save along with this array.  Ignored by non-FITS
            implementations.
        start, optional
            Logical pixel indexes for the first element in ``array`` in all
            dimensions.
        add_wcs_default, optional
            Whether to include a FITS WCS in the header for this HDU, if
            writing to FITS, and a WCS is available from a sibling or parent
            object (see `export_fits_header`) and `FitsOptions.add_wcs` is
            `None`.

        Returns
        -------
        array_model
            A Pydantic model that either stores array values inline or points
            to an out-of-tree storage location.
        """
        raise NotImplementedError()
