from __future__ import annotations

__all__ = ("Image", "ImageFieldInfo")

from collections.abc import Callable
from typing import Any, Literal
import pydantic
import pydantic_core.core_schema as pcs

import numpy as np
import numpy.typing as npt


from ._dtypes import Unit, dtype_to_str, NumberType
from ._geom import Box, Point
from ._yaml import YamlModel
from ._field_base import FieldInfoBase


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
        result = ArrayReference(
            source="TODO!", shape=self.bbox.size.shape, datatype=dtype_to_str(self.array.dtype, NumberType)
        )
        if self.unit is not None:
            return ImageReference(
                data=QuantityArrayReference(value=result, unit=self.unit),
                start=self.bbox.start,
                _serialize_extra=self._get_array,
            )
        return ImageReference(data=result, start=self.bbox.start, _serialize_extra=self._get_array)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        result = handler(ImageReference.__pydantic_core_schema__)
        result["shoefits"] = {"field_type": "image"}
        return result

    def _get_array(self) -> np.ndarray:
        return self._array


class ArrayReference(YamlModel, yaml_tag="!core/ndarray-1.1.0"):
    source: str
    shape: tuple[int, ...]
    datatype: NumberType
    byteorder: Literal["big"] = "big"


class QuantityArrayReference(YamlModel, yaml_tag="!unit/quantity-1.2.0"):
    value: ArrayReference
    unit: Unit


class ImageReference(YamlModel, yaml_tag="!shoefits/image"):
    data: QuantityArrayReference | ArrayReference
    start: Point

    _serialize_extra: Callable[[], np.ndarray]


class ImageFieldInfo(FieldInfoBase):
    field_type: Literal["image"] = "image"
    dtype: NumberType
    unit: Unit | None = None

    @property
    def is_data_export(self) -> bool:
        return True
