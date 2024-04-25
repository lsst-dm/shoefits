from __future__ import annotations

__all__ = ("Image", "image_field")

from typing import Any, Literal
import pydantic
import pydantic_core.core_schema as pcs

import numpy as np
import numpy.typing as npt


from .schema import Unit, dtype_to_number_str, NumberType
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
        result = ArrayReference(
            source="TODO!", shape=self.bbox.size.shape, datatype=dtype_to_number_str(self.array.dtype)
        )
        if self.unit is not None:
            return ImageReference(
                data=QuantityArrayReference(value=result, unit=self.unit), start=self.bbox.start
            )
        return ImageReference(data=result, start=self.bbox.start)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        result = handler(ImageReference.__pydantic_core_schema__)
        result["shoefits"] = {"schema_type": "image"}
        return result


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


def image_field(
    dtype: npt.DTypeLike,
    unit: Unit | None = None,
    **kwargs: Any,
) -> pydantic.fields.FieldInfo:
    return pydantic.Field(
        json_schema_extra={
            "shoefits": {
                "schema_type": "image",
                "dtype": dtype_to_number_str(dtype),
                "unit": unit,
            },
        },
        **kwargs,
    )
