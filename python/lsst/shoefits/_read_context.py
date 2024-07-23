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

__all__ = ("ReadContext", "ReadError")

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, final

import astropy.io.fits
import numpy as np
import pydantic

from ._geom import Box

if TYPE_CHECKING:
    from . import asdf_utils
    from ._polymorphic import PolymorphicAdapterRegistry


class ReadError(RuntimeError):
    """Exception raised when reading a serialized data structure fails."""


class ReadContext(ABC):
    """Base class for objects that read serialized data structures.

    Parameters
    ----------
    parameters, optional
        A string-keyed mapping that can be used to control what is read.  The
        ``bbox`` key is reserved as a way specify slices of larger arrays to
        be read.  Objects with custom Pydantic validators can define their own
        parameters as well.
    polymorphic_adapter_registry
        A registry used to load objects in fields annotated as `Polymorphic`.

    Notes
    -----
    `ReadContext` is designed to be used by invoking Pydantic validation
    machinery with a `ReadContext` instance included in Pydantic's "Validation
    Context" dictionary.  Subclasses are expected to have their own entry
    points that depend on how the serialized form is represented in Python.
    """

    def __init__(
        self,
        parameters: Mapping[str, Any] | None = None,
        *,
        polymorphic_adapter_registry: PolymorphicAdapterRegistry,
    ):
        self.parameters = parameters or {}
        self.polymorphic_adapter_registry = polymorphic_adapter_registry
        self._no_parameter_bbox_depth = 0

    @staticmethod
    def from_info(info: pydantic.ValidationInfo) -> ReadContext | None:
        """Extract a `ReadContext` instance inside a custom validator.

        Parameters
        ----------
        info
            The ``info`` argument passed to custom Pydantic validators.

        Returns
        -------
        read_context
            Extracted read context, or `None` if there was no validation
            context or no read context within it.
        """
        if info.context is None:
            return None
        return info.context.get("shoefits.read_context")

    def inject(self, mapping: dict[str, Any] | None = None) -> dict[str, Any]:
        """Inject the read context into a dictionary that will be passed as
        the Pydantic validation context.

        Parameters
        ----------
        mapping, optional
            If provided, an existing dictionary to modify in place and return.

        Returns
        -------
        mapping
            A mapping suitable for use as the Pydantic validation context.
        """
        if mapping is None:
            mapping = {}
        mapping["shoefits.read_context"] = self
        return mapping

    @property
    def primary_header(self) -> astropy.io.fits.Header | None:
        """The primary FITS header of the file being read, if it is available.

        This is always `None` for non-FITS serialization formats (unless they
        intentionally mimic FITS header storage) and may be `None` or empty for
        object being read within `nested` contexts (where any header keys
        written by these objects are not sent to the primary HDU).
        """
        return None

    @abstractmethod
    def nested(self) -> AbstractContextManager[None]:
        """Return a context manager for a level of logical nesting in the
        serializated form.

        Any type that uses this context manager in validation/read must use
        `WriteContext.nested` in the same way during serialization.
        """
        raise NotImplementedError()

    @abstractmethod
    def no_parameter_bbox(self) -> AbstractContextManager[None]:
        """Return a context manager that indicates that this object and any
        nested underneath it should not consider the ``bbox`` entry of the
        `parameters` mapping.
        """
        raise NotImplementedError()

    @final
    def get_parameter_bbox(self) -> Box | None:
        """Return the ``bbox`` entry in the `parameters` mapping, accounting
        for the `no_parameter_bbox` context.
        """
        if self._no_parameter_bbox_depth == 0:
            return self.parameters.get("bbox")
        return None

    @abstractmethod
    def get_array(
        self,
        array_model: asdf_utils.ArrayModel,
        bbox_from_shape: Callable[[tuple[int, ...]], Box] = Box.from_shape,
        slices_from_bbox: Callable[[Box], tuple[slice, ...]] | None = None,
    ) -> np.ndarray:
        """Read a serialized array.

        Parameters
        ----------
        array_model
            Pydantic model describing how the array was saved (may contain the
            array data itself).
        bbox_from_shape, optional
            A callback that is passed the shape of the array (which is always
            extracted from ``array_model``) and returns the full bounding box
            of the array, accounting for any nonzero origin that may have been
            stored outside ``array_model``.  The default assumes the origin
            is at coordinate 0 in all dimensions.
        slices_from_bbox, optional
            A callback that is given the full bbox of the array and returns
            slices to apply in order to read only a subset the array.  This
            is normally a closure that calls `get_parameter_bbox`; it may
            then adjust the returned bbox to reflect how it should be applied
            to a high-level object that holds an array.

        Returns
        -------
        array
            Loaded array.
        """
        raise NotImplementedError()
