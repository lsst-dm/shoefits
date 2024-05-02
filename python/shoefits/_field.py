from __future__ import annotations

__all__ = (
    "field",
    "FieldInfo",
    "FrameFieldInfo",
    "DataExportFieldInfo",
    "HeaderExportFieldInfo",
    "is_header_export",
    "is_data_export",
    "is_frame_field",
    "is_value_field",
    "is_image_field",
)


from typing import Annotated, Any, Literal, TypeAlias, TypedDict, TypeGuard, Union, cast, overload

import numpy.typing as npt
import pydantic
from pydantic.json_schema import JsonDict

from ._dtypes import Unit
from ._field_base import UnsupportedStructureError, ValueFieldInfo, make_value_field_info
from ._images import ImageFieldInfo


class FrameFieldInfo(TypedDict):
    field_type: Literal["frame"]


HeaderExportFieldInfo: TypeAlias = Annotated[
    Union[ValueFieldInfo], pydantic.Field(discriminator="field_type")
]


DataExportFieldInfo: TypeAlias = Annotated[Union[ImageFieldInfo], pydantic.Field(discriminator="field_type")]


FieldInfo: TypeAlias = Annotated[
    Union[ImageFieldInfo, ValueFieldInfo, FrameFieldInfo],
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
            # This field does not have a shoefits type.  It's a scalar
            # int/str/float we're annotating for export to FITS.
            case {"type": "integer"}:
                kwargs.setdefault("dtype", "int64")
                return make_value_field_info(**kwargs)
            case {"type": "string"}:
                kwargs.setdefault("dtype", "str")
                return make_value_field_info(**kwargs)
            case {"type": "number"}:
                kwargs.setdefault("dtype", "float64")
                return make_value_field_info(**kwargs)
            # TODO: support type unions with at least None.
        raise UnsupportedStructureError("Unsupported type for field.")


def is_header_export(field_info: FieldInfo) -> TypeGuard[HeaderExportFieldInfo]:
    return field_info["field_type"] == "value" and bool(field_info["fits_header"])


def is_data_export(field_info: FieldInfo) -> TypeGuard[DataExportFieldInfo]:
    return field_info["field_type"] == "image"


def is_frame_field(field_info: FieldInfo) -> TypeGuard[FrameFieldInfo]:
    return field_info["field_type"] == "frame"


def is_value_field(field_info: FieldInfo) -> TypeGuard[ValueFieldInfo]:
    return field_info["field_type"] == "value"


def is_image_field(field_info: FieldInfo) -> TypeGuard[ImageFieldInfo]:
    return field_info["field_type"] == "value"


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
