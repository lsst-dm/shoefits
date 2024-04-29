from __future__ import annotations

__all__ = ("frame_field", "ListFieldInfo", "DictFieldInfo", "FieldInfo")


from typing import cast, TypeAlias, Annotated, Union, Literal, Any

import pydantic
from pydantic.json_schema import JsonDict

from ._field_base import UnsupportedStructureError, ValueFieldInfo, FieldInfoBase
from ._images import ImageFieldInfo


class ListFieldInfo(FieldInfoBase):
    field_type: Literal["list"] = "list"
    item_info: FieldInfo


class DictFieldInfo(FieldInfoBase):
    field_type: Literal["dict"] = "dict"
    value_info: FieldInfo


FieldInfo: TypeAlias = Annotated[
    Union[ImageFieldInfo, ValueFieldInfo, ListFieldInfo, DictFieldInfo],
    pydantic.Field(discriminator="field_type"),
]


class _FieldSchemaCallback:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def __call__(self, json_schema: JsonDict) -> None:
        if hasattr(self, "info"):
            json_schema["shoefits"] = self.kwargs
        else:
            self.info = _frame_field_helper.process_json_schema(json_schema, self.kwargs)
            self.kwargs = self.info.model_dump()
            json_schema["shoefits"] = self.kwargs

    info: FieldInfo


class _FieldHelper:
    def __init__(self) -> None:
        self._type_adapter: pydantic.TypeAdapter[FieldInfo] | None = None

    @property
    def type_adapter(self) -> pydantic.TypeAdapter[FieldInfo]:
        if self._type_adapter is None:
            self._type_adapter = pydantic.TypeAdapter(FieldInfo)
        return self._type_adapter

    def process_json_schema(self, type_schema: JsonDict, kwargs: dict[str, Any]) -> FieldInfo:
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


_frame_field_helper = _FieldHelper()


def frame_field(**kwargs: Any) -> pydantic.fields.FieldInfo:
    return pydantic.Field(**kwargs, json_schema_extra=_FieldSchemaCallback(**kwargs))
