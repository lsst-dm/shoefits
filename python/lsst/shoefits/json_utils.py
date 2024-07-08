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

"""Utility code for working with JSON or raw JSON-derived dictionaries."""

from __future__ import annotations

__all__ = ("JsonValue",)


from typing import TypeAlias, Union

JsonValue: TypeAlias = Union[int, str, float, None, list["JsonValue"], dict[str, "JsonValue"]]
