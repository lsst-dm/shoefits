from __future__ import annotations

import pydantic.json

__all__ = ("Frame",)

from typing import Any, Self

import pydantic

from ._yaml import YamlModel


class Frame(YamlModel, yaml_tag="!shoefits/frame-0.0.1"):
    model_config = pydantic.ConfigDict(json_schema_extra={"shoefits": {"field_type": "frame"}})

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
