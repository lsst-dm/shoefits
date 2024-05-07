from __future__ import annotations

__all__ = ("Mask",)

import dataclasses
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from ._asdf import NdArray
from ._dtypes import UnsignedIntegerType, numpy_to_str
from ._geom import Box, Point
from ._yaml import YamlModel


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
        stride = self._dtype.itemsize
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
            mask = self._bits[plane]
            result[mask.index] |= mask
        return result


class Mask:
    def __init__(self, array: np.ndarray, bbox: Box, schema: MaskSchema):
        self._array = array
        self._bbox = bbox
        self._schema = schema

    @classmethod
    def from_zeros(cls, dtype: npt.DTypeLike, bbox: Box, schema: MaskSchema) -> Mask:
        return cls(np.zeros(bbox.size.shape, dtype=dtype), bbox, schema)

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
        raise NotImplementedError()

    def _serialize(self, info: pydantic.SerializationInfo) -> MaskReference:
        result = NdArray(
            source="TODO!",
            shape=self.bbox.size.shape + (self._schema.mask_size,),
            datatype=numpy_to_str(self.array.dtype, UnsignedIntegerType),
        )
        return MaskReference(
            data=result, start=self.bbox.start, planes=list(self._schema), _serialize_extra=self._get_array
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(MaskReference.__pydantic_core_schema__)

    def _get_array(self) -> np.ndarray:
        return self._array


class MaskReference(YamlModel, yaml_tag="!shoefits/mask-0.0.1"):
    data: NdArray
    start: Point
    planes: list[MaskPlane | None]

    _serialize_extra: Callable[[], np.ndarray]