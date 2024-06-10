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
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import numpy as np
import pydantic

if TYPE_CHECKING:
    from . import asdf_utils
    from ._geom import Point
    from ._polymorphic import PolymorphicAdapterRegistry


class ReadError(RuntimeError):
    pass


class ReadContext(ABC):
    @staticmethod
    def from_info(info: pydantic.ValidationInfo) -> ReadContext | None:
        if info.context is None:
            return None
        return info.context.get("shoefits.read_context")

    def inject(self, mapping: dict[str, Any] | None = None) -> dict[str, Any]:
        if mapping is None:
            mapping = {}
        mapping["shoefits.read_context"] = self
        return mapping

    @property
    @abstractmethod
    def polymorphic_adapter_registry(self) -> PolymorphicAdapterRegistry:
        raise NotImplementedError()

    @abstractmethod
    def subheader(self) -> AbstractContextManager[None]:
        raise NotImplementedError()

    @abstractmethod
    def no_parameter_bbox(self) -> AbstractContextManager[None]:
        raise NotImplementedError()

    @abstractmethod
    def get_array(self, array_model: asdf_utils.ArrayModel, start: Point) -> np.ndarray:
        raise NotImplementedError()
