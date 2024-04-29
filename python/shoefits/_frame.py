from __future__ import annotations

import pydantic.json

__all__ = ("Frame",)

from typing import Any, ClassVar, Self, Literal

import pydantic

from ._yaml import YamlModel
from ._schema import Schema
from ._field_base import FieldInfoBase


class Frame(YamlModel, yaml_tag="!shoefits/frame-0.0.1"):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls.frame_schema = Schema.build(cls)

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _frame_validate(
        cls, data: Any, handler: pydantic.ValidatorFunctionWrapHandler, info: pydantic.ValidationInfo
    ) -> Self:
        # TODO: Extract export pointer entries from data, load them using
        # validation context, replace them with full objects.
        return handler(data)

    def _serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> dict[str, Any]:
        return handler(self)

    frame_schema: ClassVar[Schema]


class FrameFieldInfo(FieldInfoBase):
    field_type: Literal["frame"] = "frame"

    @property
    def is_frame(self) -> bool:
        return True
