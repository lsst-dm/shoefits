from __future__ import annotations

__all__ = ("FieldInfoBase", "UnsupportedStructureError", "ValueFieldInfo")

from typing import Literal
import pydantic
from ._dtypes import Unit, ValueType


class UnsupportedStructureError(NotImplementedError):
    pass


class FieldInfoBase(pydantic.BaseModel):
    field_type: str

    @property
    def is_header_export(self) -> bool:
        return False

    @property
    def is_data_export(self) -> bool:
        return False

    @property
    def is_frame(self) -> bool:
        return False


class ValueFieldInfo(FieldInfoBase):
    field_type: Literal["value"] = "value"
    dtype: ValueType
    unit: Unit | None = None
    fits_header: str | bool = False

    @property
    def is_header_export(self) -> bool:
        return bool(self.fits_header)
