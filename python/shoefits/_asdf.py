from __future__ import annotations

__all__ = (
    "NdArray",
    "Quantity",
    "BlockWriter",
    "Unit",
)

from typing import Annotated, Any, BinaryIO, ClassVar, Literal, cast

import astropy.units
import numpy as np
import pydantic
import pydantic_core.core_schema as pcs
import yaml

from ._dtypes import NumberType
from ._yaml import DeferredYaml, RestrictedYamlLoader, YamlModel, yaml_represent_str


class BlockWriter:
    def __init__(self) -> None:
        self._arrays: list[np.ndarray] = []

    def add_array(self, array: np.ndarray) -> int:
        source = len(self._arrays)
        self._arrays.append(array)
        return source

    def write(self, buffer: BinaryIO) -> None:
        if self._arrays:
            raise NotImplementedError("TODO")


class _UnitAnnotation:
    TAG: ClassVar[str] = "!unit/unit-1.0.0"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_str_schema = pcs.chain_schema(
            [pcs.str_schema(), pcs.no_info_plain_validator_function(astropy.units.Unit)]
        )
        return pcs.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(astropy.units.Unit), from_str_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize, info_arg=True),
        )

    @classmethod
    def _serialize(cls, instance: astropy.units.Unit, info: pydantic.SerializationInfo) -> DeferredYaml:
        s = instance.to_string(format="vounit")
        if info.context is not None and info.context.get("yaml"):
            return DeferredYaml(yaml_represent_str, s, cls.TAG)
        return s

    @classmethod
    def _construct_yaml(
        cls, loader: yaml.Loader | yaml.FullLoader | yaml.SafeLoader, node: yaml.Node
    ) -> YamlModel:
        return astropy.units.Unit(loader.construct_scalar(cast(yaml.ScalarNode, node)))


RestrictedYamlLoader.add_constructor(_UnitAnnotation.TAG, _UnitAnnotation._construct_yaml)


Unit = Annotated[astropy.units.Unit, _UnitAnnotation]


class NdArray(YamlModel, yaml_tag="!core/ndarray-1.1.0"):
    source: str | int
    shape: tuple[int, ...]
    datatype: NumberType
    byteorder: Literal["big"] = "big"


class Quantity(YamlModel, yaml_tag="!unit/quantity-1.2.0"):
    value: NdArray
    unit: Unit
