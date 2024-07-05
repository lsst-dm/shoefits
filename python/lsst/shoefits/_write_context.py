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
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import astropy.io.fits
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
    def get_fits_write_options(self) -> FitsOptions | None:
        """Return the current `FitsOptions`.

        Non-FITS implementations should return `None`.
        """
        raise NotImplementedError()

    @abstractmethod
    def fits_write_options(self, options: FitsOptions) -> AbstractContextManager[None]:
        """Return a context manager that applies FITS-specific write options
        to this object any any nested under it.

        When `FitsWriteOptions.subheader` is `True`, this should be paired with
        a call to `ReadContext.subheader` in the corresponding validation
        logic.
        """
        raise NotImplementedError()

    @abstractmethod
    def export_header_key(
        self,
        key: str,
        value: int | str | float | bool,
        comment: str | None = None,
        hierarch: bool = False,
    ) -> None:
        """Export a FITS header entry.

        Parameters
        ----------
        key
            Header key.
        value
            Header value
        comment, optional
            Description to include the on the same line as the key and value;
            should be short enough to fit all within 80 characters.
        hierarch, optional
            Use the HIERARCH convention.  If this is `False`, ``key`` must be
            all caps and 8 characters or less.

        Notes
        -----
        Implementations that do not write to FITS should do nothing.
        """
        raise NotImplementedError()

    @abstractmethod
    def export_header_update(self, header: astropy.io.fits.Header, for_read: bool = False) -> None:
        """Export multiple FITS header entries.

        Parameters
        ----------
        header
            Header entries to add.
        for_read, optional
            If `True`, write header entries with the expectation that they need
            to be read back in later (which is otherwise not usually the case;
            generally we prefer to duplicate header information in a JSON or
            YAML tree).  This is only guaranteed to work if there is no
            subheader nesting in the context (see `fits_write_options`) when
            this is called.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_array(self, array: np.ndarray, header: astropy.io.fits.Header | None = None) -> ArrayModel:
        """Write an array.

        Parameters
        ----------
        array
            Array to save.
        header, optional
            Header entries to save along with this array.  Ignored by non-FITS
            implementations.

        Returns
        -------
        array_model
            A Pydantic model that either stores array values inline or points
            to an out-of-tree storage location.
        """
        raise NotImplementedError()
