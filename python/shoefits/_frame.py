from __future__ import annotations

import pydantic.json

__all__ = ("Frame",)

from typing import Any, ClassVar, Self

import pydantic
import pydantic_core.core_schema as pcs

from ._yaml import YamlModel
from ._field import FrameFieldInfo
from ._schema import Schema


class Frame(YamlModel, yaml_tag="!shoefits/frame-0.0.1"):
    model_config = pydantic.ConfigDict(json_schema_extra={"shoefits": {"field_type": "field"}})

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

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        result = handler(_core_schema)
        if hasattr(cls, "frame_schema"):
            result["shoefits"] = FrameFieldInfo(
                field_type="frame",
                header_exports=list(cls.frame_schema.header_exports),
                data_exports=list(cls.frame_schema.data_exports),
                children=[str(path) for path in cls.frame_schema.children],
            )
        else:
            result["shoefits"] = FrameFieldInfo(
                field_type="frame",
                header_exports=[],
                data_exports=[],
                children=[],
            )
        return result

    frame_schema: ClassVar[Schema]
