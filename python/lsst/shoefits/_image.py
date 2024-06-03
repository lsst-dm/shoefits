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

__all__ = ("Image", "ImageReference")

from typing import Any, cast

import astropy.units
import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from . import asdf_utils
from ._geom import Box, Extent, Point


class Image:
    def __init__(
        self,
        array_or_fill: np.ndarray | int | float = 0,
        /,
        *,
        bbox: Box | None = None,
        start: Point = Point(x=0, y=0),
        size: Extent | None = None,
        unit: astropy.units.Unit | None = None,
        dtype: npt.DTypeLike | None = None,
    ):
        if isinstance(array_or_fill, np.ndarray):
            if dtype is not None:
                array = np.array(array_or_fill, dtype=dtype)
            else:
                array = array_or_fill
            if bbox is None:
                bbox = Box.from_size(Extent.from_shape(cast(tuple[int, int], array.shape)), start=start)
            elif bbox.size.shape != array.shape:
                raise ValueError(
                    f"Explicit bbox size {bbox.size} does not match array with shape {array.shape}."
                )
            if size is not None and size.shape != array.shape:
                raise ValueError(f"Explicit size {size} does not match array with shape {array.shape}.")

        else:
            if bbox is None:
                if size is None:
                    raise TypeError("No bbox, size, or array provided.")
                bbox = Box.from_size(size, start=start)
            array = np.full(bbox.size.shape, array_or_fill, dtype=dtype)
        self._array = array
        self._bbox = bbox
        self._unit = unit

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        self._array[:, :] = value

    @property
    def unit(self) -> astropy.units.Unit | None:
        return self._unit

    @property
    def bbox(self) -> Box:
        return self._bbox

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_ref_schema = pcs.chain_schema(
            [
                ImageReference.__pydantic_core_schema__,
                pcs.with_info_plain_validator_function(cls._from_reference),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_ref_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Image), from_ref_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def _from_reference(cls, reference: ImageReference, info: pydantic.ValidationInfo) -> Image:
        array_model, unit = reference.unpack()
        array = asdf_utils.ArraySerialization.from_model(array_model, info)
        return cls(array, start=reference.start, unit=unit)

    def _serialize(self, info: pydantic.SerializationInfo) -> ImageReference:
        data = asdf_utils.ArraySerialization.serialize(self.array, info)
        return ImageReference.pack(data, self.bbox.start, self.unit)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(ImageReference.__pydantic_core_schema__)


class ImageReference(pydantic.BaseModel):
    data: asdf_utils.QuantityModel | asdf_utils.ArrayModel
    start: Point

    @classmethod
    def pack(
        cls, array_model: asdf_utils.ArrayModel, start: Point, unit: asdf_utils.Unit | None
    ) -> ImageReference:
        if unit is None:
            return cls.model_construct(data=array_model, start=start)
        return cls.model_construct(
            data=asdf_utils.QuantityModel.model_construct(data=array_model, unit=unit), start=start
        )

    def unpack(self) -> tuple[asdf_utils.ArrayModel, asdf_utils.Unit | None]:
        if isinstance(self.data, asdf_utils.QuantityModel):
            if not isinstance(self.data.value, asdf_utils.ArrayReferenceModel | asdf_utils.InlineArrayModel):
                raise ValueError("Expected array quantity, not scalar.")
            return self.data.value, self.data.unit
        return self.data, None
