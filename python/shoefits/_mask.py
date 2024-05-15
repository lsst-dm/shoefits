from __future__ import annotations

__all__ = ("Mask", "MaskPlane", "MaskSchema", "MaskReference")

import dataclasses
import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from . import asdf_utils
from ._dtypes import UnsignedIntegerType, numpy_to_str
from ._field_info import MaskFieldInfo
from ._geom import Box, Point


@dataclasses.dataclass(frozen=True)
class MaskPlane:
    name: str
    description: str


MaskFieldInfo.model_rebuild()


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
    def __init__(self, array: np.ndarray, bbox: Box, schema: MaskSchema):
        self._array = array
        self._bbox = bbox
        self._schema = schema

    @classmethod
    def from_zeros(cls, dtype: npt.DTypeLike, bbox: Box, schema: MaskSchema) -> Mask:
        return cls(np.zeros(bbox.size.shape + (schema.mask_size,), dtype=dtype), bbox, schema)

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
        if info.context is None or "block_writer" not in info.context:
            raise NotImplementedError("Inline arrays not yet supported.")
        writer: asdf_utils.BlockWriter = info.context["block_writer"]
        return MaskReference.from_mask_and_source(self, writer.add_array(self.array))

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(MaskReference.__pydantic_core_schema__)

    def _get_array(self) -> np.ndarray:
        return self._array


class MaskReference(pydantic.BaseModel):
    data: asdf_utils.NdArray
    start: Point
    planes: list[MaskPlane | None]
    address: int | None = None

    @classmethod
    def from_mask_and_source(cls, mask: Mask, source: str | int) -> Self:
        result = asdf_utils.NdArray(
            source=source,
            shape=mask.bbox.size.shape + (mask._schema.mask_size,),
            datatype=numpy_to_str(mask.array.dtype, UnsignedIntegerType),
        )
        return cls(data=result, start=mask.bbox.start, planes=list(mask._schema))

    @pydantic.model_serializer(mode="wrap")
    def _serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> dict[str, Any]:
        result = handler(self)
        if info.context is not None:
            info.context["addressed"] = result
        return result
