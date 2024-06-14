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
    "MaskHeaderFormat",
    "FitsOptions",
    "ExportFitsHeaderKey",
    "fits_header_exporter",
)

import dataclasses
import enum
from collections.abc import Callable
from typing import Any, TypeVar

import astropy.io.fits
import pydantic
import pydantic_core.core_schema as pcs

from ._geom import Extent
from ._read_context import ReadContext
from ._write_context import WriteContext


class FitsCompressionAlgorithm(enum.Enum):
    GZIP_1 = "GZIP_1"
    GZIP_2 = "GZIP_2"


class MaskHeaderFormat(enum.Enum):
    AFW = "afw"


@dataclasses.dataclass
class FitsCompression:
    tile_size: Extent
    algorithm: FitsCompressionAlgorithm = FitsCompressionAlgorithm.GZIP_2


@dataclasses.dataclass
class FitsOptions:
    extname: str | None = None
    mask_header_style: MaskHeaderFormat | None = None
    compression: FitsCompression | None = None
    subheader: bool = False

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        base_schema = pcs.with_info_wrap_validator_function(self.validate, handler(source_type))
        return pcs.json_or_python_schema(
            json_schema=base_schema,
            python_schema=base_schema,
            serialization=pcs.wrap_serializer_function_ser_schema(self.serialize, info_arg=True),
        )

    def validate(
        self, data: Any, handler: pydantic.ValidatorFunctionWrapHandler, info: pydantic.ValidationInfo
    ) -> Any:
        if self.subheader and (read_context := ReadContext.from_info(info)):
            with read_context.subheader():
                return handler(data)
        return handler(data)

    def serialize(
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


class ExportFitsHeaderKey:
    def __init__(self, key: str, hierarch: bool = False, comment: str = ""):
        self.key = key
        self.hierarch = hierarch
        self.comment = comment

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        base_schema = handler(source_type)
        return pcs.json_or_python_schema(
            json_schema=base_schema,
            python_schema=base_schema,
            serialization=pcs.wrap_serializer_function_ser_schema(self.serialize, info_arg=True),
        )

    def serialize(
        self,
        obj: Any,
        handler: pydantic.SerializerFunctionWrapHandler,
        info: pydantic.SerializationInfo,
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            write_context.export_header_key(self.key, obj, comment=self.comment, hierarch=self.hierarch)
        return handler(obj)


_T = TypeVar("_T")


def fits_header_exporter(func: Callable[[_T], astropy.io.fits.Header]) -> Any:
    @pydantic.model_serializer(mode="wrap")
    def _fits_header_export_serializer(
        self: _T, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> Any:
        if write_context := WriteContext.from_info(info):
            write_context.export_header_update(func(self))
        return handler(self)

    return _fits_header_export_serializer
