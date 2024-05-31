# This file is part of lsst-shoefits.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

__all__ = ()

from abc import ABC, abstractmethod
from typing import Annotated

import astropy.units as u
import jsonschema
import numpy as np
import pydantic

import lsst.shoefits as shf


class Thing(ABC):
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError()


@shf.register_tag("A")
class ThingA(Thing):
    def __init__(self, i: int):
        self.i = i

    def value(self) -> int:
        return self.i


class SerializedA(pydantic.BaseModel):
    data: int = 0


class AdapterA(shf.PolymorphicAdapter[ThingA, SerializedA]):
    @property
    def model_type(self) -> type[SerializedA]:
        return SerializedA

    def to_model(self, polymorphic: ThingA) -> SerializedA:
        return SerializedA(data=polymorphic.i)

    def from_model(self, serialized: SerializedA) -> ThingA:
        return ThingA(i=serialized.data)


@shf.register_tag("B")
class ThingB(pydantic.BaseModel, Thing):
    data: int = 0

    def value(self) -> int:
        return self.data


adapter_registry = shf.PolymorphicAdapterRegistry()
adapter_registry.register_adapter("A", AdapterA())
adapter_registry.register_native("B", ThingB)


class Example(pydantic.BaseModel):
    array: shf.asdf_utils.Array
    unit: shf.asdf_utils.Unit
    thing: Annotated[Thing, shf.Polymorphic()] | None = None


def test_round_trip_json_inline() -> None:
    """Test that the Example class round-trips through Pydantic JSON
    serialization when no array writer is provided.
    """
    context = dict(polymorphic_adapter_registry=adapter_registry)
    e1 = Example(array=np.random.randn(3, 4), unit=u.s, thing=ThingA(i=2))
    s = e1.model_dump_json(context=context)
    e2 = Example.model_validate_json(s, context=context)
    np.testing.assert_array_equal(e1.array, e2.array)
    assert e1.unit == e2.unit
    assert isinstance(e2.thing, ThingA)
    assert e2.thing.value() == 2


def test_round_trip_python_inline() -> None:
    """Test that the Example class round-trips through Pydantic Python-object
    serialization when no array writer is provided.
    """
    context = dict(polymorphic_adapter_registry=adapter_registry)
    e1 = Example(array=np.random.randn(3, 4), unit=u.s, thing=ThingB(data=3))
    d = e1.model_dump(context=context)
    e2 = Example.model_validate(d, context=context)
    np.testing.assert_array_equal(e1.array, e2.array)
    assert e1.unit == e2.unit
    assert isinstance(e2.thing, ThingB)
    assert e2.thing.value() == 3


def test_schema_inline() -> None:
    """Test that the Example class JSON schema matches its serialization,
    and that the schema has the special information we inject.
    """
    context = dict(polymorphic_adapter_registry=adapter_registry)
    schema = Example.model_json_schema()
    assert "http://stsci.edu" in schema["properties"]["array"]["$schema"]
    assert "http://stsci.edu" in schema["properties"]["unit"]["$schema"]
    assert "asdf/core/ndarray-1.1.0" in schema["properties"]["array"]["id"]
    assert "asdf/unit/unit-1.0.0" in schema["properties"]["unit"]["id"]
    assert schema["properties"]["thing"]["anyOf"] == [{"type": "object"}, {"type": "null"}]
    jsonschema.validate(Example(array=np.random.randn(3, 4), unit=u.s).model_dump(), schema)
    jsonschema.validate(
        Example(array=np.random.randn(3, 4), unit=u.s, thing=ThingA(i=4)).model_dump(context=context),
        schema,
    )
    jsonschema.validate(
        Example(array=np.random.randn(3, 4), unit=u.s, thing=ThingB(data=5)).model_dump(context=context),
        schema,
    )
