from __future__ import annotations

__all__ = ("value_field",)

from typing import Any, Literal

import numpy.typing as npt
import pydantic

from ._dtypes import Unit, dtype_to_str, ValueType
from ._schema_base import FitsExportSchemaBase


def value_field(
    *,
    dtype: npt.DTypeLike | None = None,
    unit: Unit | None = None,
    fits_header: str | bool = False,
    **kwargs: Any,
) -> pydantic.fields.FieldInfo:
    return pydantic.Field(
        json_schema_extra={
            "shoefits": {
                "export_type": "value",
                "dtype": dtype_to_str(dtype, ValueType),
                "unit": unit,
                "fits_header": fits_header,
            }
        },
        **kwargs,
    )


class ValueSchema(FitsExportSchemaBase):
    export_type: Literal["value"] = "value"
    dtype: ValueType
    unit: Unit | None
    fits_header: str | bool = False

    @property
    def is_header_export(self) -> bool:
        return bool(self.fits_header)
