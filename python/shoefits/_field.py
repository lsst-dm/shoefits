from __future__ import annotations

__all__ = ("field", "ListFieldInfo", "DictFieldInfo", "FieldInfo")


from typing import cast, TypeAlias, Annotated, Union, Literal, Any, TypedDict, overload

import numpy.typing as npt
import pydantic
from pydantic.json_schema import JsonDict

from ._dtypes import Unit
from ._field_base import UnsupportedStructureError, ValueFieldInfo, make_value_field_info
from ._images import ImageFieldInfo


class ListFieldInfo(TypedDict):
    field_type: Literal["list"]
    item_info: FieldInfo


class DictFieldInfo(TypedDict):
    field_type: Literal["dict"]
    value_info: FieldInfo


class FrameFieldInfo(TypedDict):
    field_type: Literal["frame"]
    header_exports: list[str]
    data_exports: list[str]
    children: list[str]


FieldInfo: TypeAlias = Annotated[
    Union[ImageFieldInfo, ValueFieldInfo, ListFieldInfo, DictFieldInfo, FrameFieldInfo],
    pydantic.Field(discriminator="field_type"),
]


class _FieldSchemaCallback:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def __call__(self, json_schema: JsonDict) -> None:
        if hasattr(self, "info"):
            json_schema["shoefits"] = self.kwargs
        else:
            self.info = _field_helper.process_json_schema(json_schema, self.kwargs)
            # MyPy doesn't think TypedDict should convert to dict[str, Any]!
            self.kwargs = self.info  # type: ignore[assignment]
            json_schema["shoefits"] = self.info  # type: ignore[assignment]

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
                return make_value_field_info(**kwargs)
            case {"type": "string"}:
                kwargs.setdefault("dtype", "str")
                return make_value_field_info(**kwargs)
            case {"type": "number"}:
                kwargs.setdefault("dtype", "float64")
                return make_value_field_info(**kwargs)
            case {"type": "array", "items": item_type_schema}:
                # Interpret kwargs as part of list item schema.
                item_info = self.process_json_schema(cast(JsonDict, item_type_schema), kwargs)
                return ListFieldInfo(item_info=item_info, field_type="list")
            case {"type": "object", "additionalProperties": value_type_schema}:
                # Interpret kwargs as part of dict value schema.
                value_info = self.process_json_schema(cast(JsonDict, value_type_schema), kwargs)
                return DictFieldInfo(value_info=value_info, field_type="dict")
            # TODO: support type unions with at least None.
        raise UnsupportedStructureError("Unsupported type for field.")

    def is_header_export(self, field_info: FieldInfo) -> bool:
        return field_info["field_type"] == "value" and bool(field_info["fits_header"])

    def is_data_export(self, field_info: FieldInfo) -> bool:
        return field_info["field_type"] == "image"

    def is_frame(self, field_info: FieldInfo) -> bool:
        return field_info["field_type"] == "frame"


_field_helper = _FieldHelper()


# Overload for ImageFieldInfo (or list/dict thereof).
@overload
def field(*, dtype: npt.DTypeLike, unit: Unit | None = None) -> pydantic.fields.FieldInfo: ...


# Overload for ValueFieldInfo (or list/dict thereof).
@overload
def field(
    *, dtype: npt.DTypeLike | None = None, unit: Unit | None = None, fits_header: bool | str = False
) -> pydantic.fields.FieldInfo: ...


def field(**kwargs: Any) -> pydantic.fields.FieldInfo:
    return pydantic.Field(**kwargs, json_schema_extra=_FieldSchemaCallback(**kwargs))
