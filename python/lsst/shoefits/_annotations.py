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
from typing import Any

import pydantic

from ._fits_options import FitsDataOptions
from ._read_context import ReadContext
from ._write_context import WriteContext


def serialize_frame(
    obj: Any, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
) -> Any:
    if write_context := WriteContext.from_info(info):
        with write_context.frame():
            return handler(obj)
    else:
        return handler(obj)


def validate_frame(
    data: Any, handler: pydantic.ValidatorFunctionWrapHandler, info: pydantic.ValidationInfo
) -> Any:
    if read_context := ReadContext.from_info(info):
        with read_context.frame():
            return handler(data)
    else:
        return handler(data)


@dataclasses.dataclass
class WithFitsDataOptions:
    def __init__(self, **kwargs: Any):
        self.options = FitsDataOptions(**kwargs)

    def __call__(
        self,
        obj: Any,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            with write_context.fits_data_options(self.options):
                return handler(obj)
        else:
            return handler(obj)


@dataclasses.dataclass
class ExportFitsHeaderKey:
    def __init__(self, key: str, hierarch: bool = False, comment: str = ""):
        self.key = key
        self.hierarch = hierarch
        self.comment = comment

    def __call__(
        self,
        obj: Any,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            write_context.export_header_key(self.key, obj, comment=self.comment, hierarch=self.hierarch)
        return handler(obj)
