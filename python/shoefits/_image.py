from __future__ import annotations

__all__ = ("Image",)

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from ._asdf import NdArray, Quantity
from ._dtypes import NumberType, Unit, numpy_to_str
from ._geom import Box, Point
from ._yaml import YamlModel


class Image:
    def __init__(self, array: np.ndarray, bbox: Box, unit: Unit | None = None):
        self._array = array
        self._bbox = bbox
        self._unit = unit

    @classmethod
    def from_zeros(cls, dtype: npt.DTypeLike, bbox: Box, unit: Unit | None = None) -> Image:
        return cls(np.zeros(bbox.size.shape, dtype=dtype), bbox, unit)

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        self._array[:, :] = value

    @property
    def unit(self) -> Unit | None:
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
        raise NotImplementedError()

    def _serialize(self, info: pydantic.SerializationInfo) -> ImageReference:
        result = NdArray(
            source="TODO!", shape=self.bbox.size.shape, datatype=numpy_to_str(self.array.dtype, NumberType)
        )
        if self.unit is not None:
            return ImageReference(
                data=Quantity(value=result, unit=self.unit),
                start=self.bbox.start,
                _serialize_extra=self._get_array,
            )
        return ImageReference(data=result, start=self.bbox.start, _serialize_extra=self._get_array)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(ImageReference.__pydantic_core_schema__)

    def _get_array(self) -> np.ndarray:
        return self._array


class ImageReference(YamlModel, yaml_tag="!shoefits/image-0.0.1"):
    data: Quantity | NdArray
    start: Point

    _serialize_extra: Callable[[], np.ndarray]
