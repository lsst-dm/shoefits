from __future__ import annotations

import pydantic.json

__all__ = ("Frame", "frame_field")

from collections.abc import Mapping
from typing import Any, ClassVar, Self, Union, TypeAlias, Annotated, cast, Literal

import pydantic
from pydantic.json_schema import JsonDict

from ._yaml import YamlModel
from ._images import ImageFieldInfo
from ._field_base import UnsupportedStructureError, ValueFieldInfo, FrameFieldInfoBase


class ListFieldInfo(FrameFieldInfoBase):
    field_type: Literal["list"] = "list"
    item_info: FrameFieldInfo


class DictFieldInfo(FrameFieldInfoBase):
    field_type: Literal["dict"] = "dict"
    value_info: FrameFieldInfo


FrameFieldInfo: TypeAlias = Annotated[
    Union[ImageFieldInfo, ValueFieldInfo, ListFieldInfo, DictFieldInfo],
    pydantic.Field(discriminator="field_type"),
]


def frame_field(**kwargs: Any) -> pydantic.fields.FieldInfo:
    return pydantic.Field(**kwargs, json_schema_extra=_FrameFieldSchemaCallback(**kwargs))


class Frame(YamlModel, yaml_tag="!shoefits/frame-0.0.1"):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # Generating and throwing away the JSON SChema is a convenient but
        # rather ugly way to combine the type-specific JSON Schema with the
        # stuff injected by frame_field, and then validate the combination.
        # It may not work with recursive or generic models, and it'd probably
        # be better to make frame_fields a lazily-evaluated mapping somehow.
        super().model_json_schema()
        cls.frame_fields = {
            name: field_info.json_schema_extra.info
            for name, field_info in cls.model_fields.items()
            if isinstance(field_info.json_schema_extra, _FrameFieldSchemaCallback)
        }

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

    frame_fields: ClassVar[Mapping[str, FrameFieldInfo]]


class _FrameFieldSchemaCallback:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.info: FrameFieldInfo

    def __call__(self, json_schema: JsonDict) -> None:
        if hasattr(self, "info"):
            json_schema["shoefits"] = self.kwargs
        else:
            self.info = _frame_field_helper.process_json_schema(json_schema, self.kwargs)
            self.kwargs = self.info.model_dump()
            json_schema["shoefits"] = self.kwargs


class _FrameFieldHelper:
    def __init__(self) -> None:
        self._type_adapter: pydantic.TypeAdapter[FrameFieldInfo] | None = None

    @property
    def type_adapter(self) -> pydantic.TypeAdapter[FrameFieldInfo]:
        if self._type_adapter is None:
            self._type_adapter = pydantic.TypeAdapter(FrameFieldInfo)
        return self._type_adapter

    def process_json_schema(self, type_schema: JsonDict, kwargs: dict[str, Any]) -> FrameFieldInfo:
        if "shoefits" in type_schema:
            field_schema = cast(JsonDict, type_schema["shoefits"])
        else:
            field_schema = {}
        if "field_type" in field_schema:
            # This field has a shoefits type (e.g. Image) that has injected
            # the right field_type discriminator value into the JSON Schema
            # already.  Override type-level defaults with kwargs and we're
            # done.
            field_schema.update(kwargs)
            return self.type_adapter.validate_python(field_schema)
        match type_schema:
            # This field does not have a shoefits type.  It's either a scalar
            # int/str/float we're annotating for export to FITS, or a list or
            # dict of shoefits types.  Inspect the JSON Schema to see which.
            case {"type": "integer"}:
                kwargs.setdefault("dtype", "int64")
                return ValueFieldInfo.model_validate(kwargs)
            case {"type": "string"}:
                kwargs.setdefault("dtype", "str")
                return ValueFieldInfo.model_validate(kwargs)
            case {"type": "number"}:
                kwargs.setdefault("dtype", "float64")
                return ValueFieldInfo.model_validate(kwargs)
            case {"type": "array", "items": item_type_schema}:
                # Interpret kwargs as part of list item schema.
                item_info = self.process_json_schema(cast(JsonDict, item_type_schema), kwargs)
                return ListFieldInfo.model_construct(item_info=item_info)
            case {"type": "object", "additionalProperties": value_type_schema}:
                # Interpret kwargs as part of dict value schema.
                value_info = self.process_json_schema(cast(JsonDict, value_type_schema), kwargs)
                return DictFieldInfo.model_construct(value_info=value_info)
            # TODO: support type unions with at least None.
        raise UnsupportedStructureError("Unsupported type for frame field.")


_frame_field_helper = _FrameFieldHelper()
