from __future__ import annotations

__all__ = ("ImageSchema", "MaskSchema", "ImagePixelType", "MaskPixelType")


from typing import Literal, TypeAlias

import pydantic

from ._frames import FrameSchema
from ._simple import IntegerType, Unit, FloatType, UnsignedIntegerType, SchemaBase


ImagePixelType: TypeAlias = IntegerType | FloatType

MaskPixelType: TypeAlias = UnsignedIntegerType


class ImageSchema(SchemaBase):
    schema_type: Literal["image"] = "image"
    pixel_type: ImagePixelType
    unit: Unit | None = None
    frame: FrameSchema | None = None


class MaskSchema(SchemaBase):
    schema_type: Literal["mask"] = "mask"
    # {name: docs}; enumerate(planes) sets bit IDs.
    planes: dict[str, str] = pydantic.Field(default_factory=dict)
    pixel_type: UnsignedIntegerType = "uint8"
    frame: FrameSchema | None = None


_ImageLikeSchemaType: TypeAlias = Literal["image", "mask"]
_ImageLikeSchema: TypeAlias = ImageSchema | MaskSchema
