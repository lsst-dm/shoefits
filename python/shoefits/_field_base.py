from __future__ import annotations

__all__ = ("FrameFieldInfoBase", "UnsupportedStructureError", "ValueFieldInfo")

from typing import Literal
import pydantic
from ._dtypes import Unit, ValueType


class UnsupportedStructureError(NotImplementedError):
    pass


class FrameFieldInfoBase(pydantic.BaseModel):
    field_type: str

    @property
    def is_nested(self) -> bool:
        return False

    @property
    def is_header_export(self) -> bool:
        return False

    @property
    def is_data_export(self) -> bool:
        return False


class ValueFieldInfo(FrameFieldInfoBase):
    field_type: Literal["value"] = "value"
    dtype: ValueType
    unit: Unit | None = None
    fits_header: str | bool = False

    @property
    def is_header_export(self) -> bool:
        return bool(self.fits_header)
