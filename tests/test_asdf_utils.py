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

import astropy.units as u
import jsonschema
import numpy as np
import pydantic
from lsst.shoefits import asdf_utils


class Example(pydantic.BaseModel):
    array: asdf_utils.Array
    unit: asdf_utils.Unit


def test_round_trip_json_inline() -> None:
    """Test that the Example class round-trips through Pydantic JSON
    serialization when no array writer is provided.
    """
    e1 = Example(array=np.random.randn(3, 4), unit=u.s)
    s = e1.model_dump_json()
    e2 = Example.model_validate_json(s)
    np.testing.assert_array_equal(e1.array, e2.array)
    assert e1.unit == e2.unit


def test_round_trip_python_inline() -> None:
    """Test that the Example class round-trips through Pydantic Python-object
    serialization when no array writer is provided.
    """
    e1 = Example(array=np.random.randn(3, 4), unit=u.s)
    d = e1.model_dump()
    e2 = Example.model_validate(d)
    np.testing.assert_array_equal(e1.array, e2.array)
    assert e1.unit == e2.unit


def test_schema_inline() -> None:
    """Test that the Example class JSON schema matches its serialization,
    and that the schema has the special information we inject.
    """
    e1 = Example(array=np.random.randn(3, 4), unit=u.s)
    schema = Example.model_json_schema()
    jsonschema.validate(e1.model_dump(), schema)
    assert "http://stsci.edu" in schema["properties"]["array"]["$schema"]
    assert "http://stsci.edu" in schema["properties"]["unit"]["$schema"]
    assert "asdf/core/ndarray-1.1.0" in schema["properties"]["array"]["id"]
    assert "asdf/unit/unit-1.0.0" in schema["properties"]["unit"]["id"]
