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
from typing import TYPE_CHECKING, Any

import numpy as np
import pydantic

from ._geom import Box

if TYPE_CHECKING:
    from . import asdf_utils
    from ._polymorphic import PolymorphicAdapterRegistry


class ReadError(RuntimeError):
    pass


class ReadContext(ABC):
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
        if info.context is None:
            return None
        return info.context.get("shoefits.read_context")

    def inject(self, mapping: dict[str, Any] | None = None) -> dict[str, Any]:
        if mapping is None:
            mapping = {}
        mapping["shoefits.read_context"] = self
        return mapping

    @abstractmethod
    def subheader(self) -> AbstractContextManager[None]:
        raise NotImplementedError()

    @abstractmethod
    def no_parameter_bbox(self) -> AbstractContextManager[None]:
        raise NotImplementedError()

    def get_parameter_bbox(self) -> Box | None:
        if self._no_parameter_bbox_depth == 0:
            return self.parameters.get("bbox")
        return None

    @abstractmethod
    def get_array(
        self,
        array_model: asdf_utils.ArrayModel,
        bbox_from_shape: Callable[[tuple[int, ...]], Box] = Box.from_shape,
        slice_result: Callable[[Box], tuple[slice, ...]] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError()
