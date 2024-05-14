from __future__ import annotations

__all__ = ("Image", "ImageReference")

from typing import Any, Self

import astropy.units
import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from . import asdf_utils
from ._dtypes import NumberType, numpy_to_str
from ._geom import Box, Point


class Image:
    def __init__(self, array: np.ndarray, bbox: Box, unit: astropy.units.Unit | None = None):
        self._array = array
        self._bbox = bbox
        self._unit = unit

    @classmethod
    def from_zeros(cls, dtype: npt.DTypeLike, bbox: Box, unit: astropy.units.Unit | None = None) -> Image:
        return cls(np.zeros(bbox.size.shape, dtype=dtype), bbox, unit)

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
        raise NotImplementedError()

    def _serialize(self, info: pydantic.SerializationInfo) -> ImageReference:
        if info.context is None or "block_writer" not in info.context:
            raise NotImplementedError("Inline arrays not yet supported.")
        writer: asdf_utils.BlockWriter = info.context["block_writer"]
        return ImageReference.from_image_and_source(self, writer.add_array(self.array))

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(ImageReference.__pydantic_core_schema__)


class ImageReference(pydantic.BaseModel):
    data: asdf_utils.Quantity | asdf_utils.NdArray
    start: Point
    address: int | None = None

    @classmethod
    def from_image_and_source(cls, image: Image, source: str | int) -> Self:
        data = asdf_utils.NdArray(
            source=source, shape=image.bbox.size.shape, datatype=numpy_to_str(image.array.dtype, NumberType)
        )
        if image.unit is not None:
            return cls(
                data=asdf_utils.Quantity(value=data, unit=image.unit),
                start=image.bbox.start,
            )
        return cls(data=data, start=image.bbox.start)

    @pydantic.model_serializer(mode="wrap")
    def _serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> dict[str, Any]:
        result = handler(self)
        if info.context is not None:
            info.context["addressed"] = result
        return result
