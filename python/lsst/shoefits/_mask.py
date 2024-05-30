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

__all__ = ("Mask", "MaskPlane", "MaskSchema", "MaskReference")

import dataclasses
import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from . import asdf_utils
from ._geom import Box, Extent, Point


@dataclasses.dataclass(frozen=True)
class MaskPlane:
    name: str
    description: str


@dataclasses.dataclass(frozen=True)
class MaskPlaneBit:
    index: int
    mask: int

    @classmethod
    def compute(cls, overall_index: int, stride: int) -> MaskPlaneBit:
        index, bit = divmod(overall_index, stride)
        return cls(index, 1 << bit)


class MaskSchema:
    def __init__(self, planes: Iterable[MaskPlane | None], dtype: npt.DTypeLike = np.uint8):
        self._planes = tuple(planes)
        self._dtype = np.dtype(dtype)
        self._descriptions = {plane.name: plane.description for plane in self._planes if plane is not None}
        stride = self._dtype.itemsize * 8
        self._mask_size = math.ceil(len(self._planes) / stride)
        self._bits: dict[str, MaskPlaneBit] = {
            plane.name: MaskPlaneBit.compute(n, stride)
            for n, plane in enumerate(self._planes)
            if plane is not None
        }

    def __iter__(self) -> Iterator[MaskPlane | None]:
        return iter(self._planes)

    def __len__(self) -> int:
        return len(self._planes)

    def __getitem__(self, i: int) -> MaskPlane | None:
        return self._planes[i]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def mask_size(self) -> int:
        return self._mask_size

    @property
    def descriptions(self) -> Mapping[str, str]:
        return self._descriptions

    def bitmask(self, *planes: str) -> np.ndarray:
        result = np.zeros(self.mask_size, dtype=self._dtype)
        for plane in planes:
            bit = self._bits[plane]
            result[bit.index] |= bit.mask
        return result


class Mask:
    def __init__(
        self,
        array_or_fill: np.ndarray | int = 0,
        /,
        *,
        bbox: Box | None = None,
        start: Point = Point(x=0, y=0),
        size: Extent | None = None,
        schema: MaskSchema,
    ):
        if isinstance(array_or_fill, np.ndarray):
            array = np.array(array_or_fill, dtype=schema.dtype)
            if bbox is None:
                bbox = Box.from_size(Extent.from_shape(cast(tuple[int, int], array.shape[:2])), start=start)
            elif bbox.size.shape + (schema.mask_size,) != array.shape:
                raise ValueError(
                    f"Explicit bbox size {bbox.size} and schema of size {schema.mask_size} do not "
                    f"match array with shape {array.shape}."
                )
            if size is not None and size.shape + (schema.mask_size,) != array.shape:
                raise ValueError(
                    f"Explicit size {size} and schema of size {schema.mask_size} do "
                    f"not match array with shape {array.shape}."
                )

        else:
            if bbox is None:
                if size is None:
                    raise TypeError("No bbox, size, or array provided.")
                bbox = Box.from_size(size, start=start)
            array = np.full(bbox.size.shape + (schema.mask_size,), array_or_fill, dtype=schema.dtype)
        self._array = array
        self._bbox = bbox
        self._schema = schema

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        self._array[:, :] = value

    @property
    def schema(self) -> MaskSchema:
        return self._schema

    @property
    def bbox(self) -> Box:
        return self._bbox

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_ref_schema = pcs.chain_schema(
            [
                MaskReference.__pydantic_core_schema__,
                pcs.with_info_plain_validator_function(cls._from_reference),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_ref_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Mask), from_ref_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def _from_reference(cls, reference: MaskReference, info: pydantic.ValidationInfo) -> Mask:
        array = asdf_utils.ArraySerialization.from_model(reference.data, info)
        schema = MaskSchema(reference.planes, dtype=array.dtype)
        return cls(array, start=reference.start, schema=schema)

    def _serialize(self, info: pydantic.SerializationInfo) -> MaskReference:
        data = asdf_utils.ArraySerialization.serialize(self.array, info)
        return MaskReference(data=data, start=self.bbox.start, planes=list(self.schema))

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(MaskReference.__pydantic_core_schema__)

    def _get_array(self) -> np.ndarray:
        return self._array


class MaskReference(pydantic.BaseModel):
    data: asdf_utils.ArrayModel
    start: Point
    planes: list[MaskPlane | None]
