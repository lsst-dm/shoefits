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

__all__ = ("Point", "Extent", "Interval", "Box", "bounds")

from typing import Any, ClassVar, cast, final, overload

import numpy as np
import pydantic


class _BaseGeometry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)


@final
class Point(_BaseGeometry):
    """A 2-d integer-valued point.

    Notes
    -----
    `Point` and `Extent` support the following operations:

    - ``point + extent -> point``
    - ``point - extent -> point``
    - ``point - point -> extent``
    - ``extent + extent -> extent``
    - ``extent - extent -> extent``

    Other permutations are either nonsensical (e.g. ``point + point``) or
    rededundant and less intuitive than the supported ones (e.g. ``extent +
    point``).
    """

    x: int
    """Column position."""

    y: int
    """Row position."""

    zero: ClassVar[Point]
    """A point with ``x=0`` and ``y=0``."""

    def __add__(self, other: Extent) -> Point:
        return Point(x=self.x + other.x, y=self.y + other.y)

    @overload
    def __sub__(self, other: Point) -> Extent: ...

    @overload
    def __sub__(self, other: Extent) -> Point: ...

    def __sub__(self, other: Point | Extent) -> Any:
        result_type = Point if type(other) is Extent else Extent
        return result_type(x=self.x - other.x, y=self.y - other.y)

    def as_extent(self) -> Extent:
        """Return an `Extent` with the same ``x`` and ``y`` values as this
        point.
        """
        return Extent(x=self.x, y=self.y)


Point.zero = Point(x=0, y=0)


@final
class Extent(_BaseGeometry):
    """A 2-d integer-valued offset vector.

    Notes
    -----
    See `Point` for supported arithmetic operations.
    """

    x: int = 0
    """Column offset."""

    y: int = 0
    """Row offset."""

    zero: ClassVar[Extent]
    """An extent with ``x=0`` and ``y=0``."""

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> Extent:
        """Construct an `Extent` from the shape of a 2-d `numpy.ndarray.

        We assume array dimensions are always ordered ``(rows, columns)``, i.e.
        ``(extent.y, extent.x)``.
        """
        return cls(x=shape[1], y=shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """The `numpy.ndarray` 2-d shape that corresponds to this extent."""
        return (self.y, self.x)

    def __add__(self, other: Extent) -> Extent:
        return Extent(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Extent) -> Extent:
        return Extent(x=self.x - other.x, y=self.y - other.y)

    def as_point(self) -> Point:
        """Return a `Point` with the same ``x`` and ``y`` values as this
        extent.
        """
        return Point(x=self.x, y=self.y)


Extent.zero = Extent(x=0, y=0)


@final
class Interval(_BaseGeometry):
    """A 1-d integer interval with positive size.

    Notes
    -----
    Adding or subtracting an `int` from an interval returns a shifted interval.
    """

    start: int
    """Inclusive minimum point in the interval."""

    stop: int
    """One past the maximum point in the interval."""

    @property
    def min(self) -> int:
        """Inclusive minimum point in the interval."""
        return self.start

    @property
    def max(self) -> int:
        """Inclusive maximum point in the interval."""
        return self.stop - 1

    @property
    def size(self) -> int:
        """Size of the interval."""
        return self.stop - self.start

    @property
    def range(self) -> range:
        """A `range` object that iterates over all values in the interval."""
        return range(self.start, self.stop)

    @property
    def arange(self) -> np.ndarray:
        """A `numpy.ndarray` of all the values in the interval."""
        return np.arange(self.start, self.stop)

    @property
    def center(self) -> float:
        """The center of the interval."""
        return 0.5 * (self.min + self.max)

    def __str__(self) -> str:
        return f"{self.start}:{self.stop}"

    @classmethod
    def hull(cls, first: int | Interval, *args: int | Interval) -> Interval:
        """Construct an interval that includes all of the given points and/or
        intervals.
        """
        if type(first) is Interval:
            rmin = first.min
            rmax = first.max
        else:
            rmin = rmax = first
        for arg in args:
            if type(arg) is Interval:
                rmin = min(rmin, arg.min)
                rmax = max(rmax, arg.max)
            else:
                rmin = min(rmin, arg)
                rmax = max(rmax, arg)
        return Interval(start=rmin, stop=rmax + 1)

    @classmethod
    def from_size(cls, size: int, start: int = 0) -> Interval:
        """Construct an interval from its size and optional start."""
        return cls(start=start, stop=start + size)

    def __add__(self, other: int) -> Interval:
        return Interval(start=self.start + other, stop=self.stop + other)

    def __sub__(self, other: int) -> Interval:
        return Interval(start=self.start - other, stop=self.stop - other)

    @pydantic.model_validator(mode="after")
    def _validate(self) -> Interval:
        if self.start > self.stop:
            raise ValueError("Intervals must have positive size.")
        return self

    def __contains__(self, x: int) -> bool:
        return x >= self.start and x < self.stop

    def contains(self, other: Interval) -> bool:
        """Test whether this interval fully contains another."""
        return self.start <= other.start and self.stop >= other.stop

    def intersection(self, other: Interval) -> Interval | None:
        """Return an interval that is contained by both ``self`` and ``other``.

        When there is no overlap between the intervals, `None` is returned.
        """
        new_start = max(self.start, other.start)
        new_stop = min(self.stop, other.stop)
        if new_start < new_stop:
            return Interval.model_construct(start=new_start, stop=new_stop)
        return None

    def slice_within(self, other: Interval) -> slice:
        """Return the `slice` that corresponds to the values of this interval
        when the items of the container being slices correspond to ``other``.
        """
        return slice(self.start - other.start, self.stop - other.start)


@final
class Box(_BaseGeometry):
    """A 2-d integer box with positive size in both dimensions.

    Notes
    -----
    Adding or subtracting an `Extent` from a box returns a shifted box.
    """

    x: Interval
    """Interval that represents the box in the column dimension."""

    y: Interval
    """Interval that represents the box in the row dimension."""

    @property
    def min(self) -> Point:
        """Inclusive minimum point in the box."""
        return Point(x=self.x.min, y=self.y.min)

    @property
    def max(self) -> Point:
        """Inclusive maximum point in the box."""
        return Point(x=self.x.max, y=self.y.max)

    @property
    def start(self) -> Point:
        """Inclusive minimum point in the box."""
        return Point(x=self.x.start, y=self.y.start)

    @property
    def stop(self) -> Point:
        """One past the maximum point in the point (in both dimensions)."""
        return Point(x=self.x.stop, y=self.y.stop)

    @property
    def size(self) -> Extent:
        """The size of the box in both dimensions."""
        return Extent(x=self.x.size, y=self.y.size)

    @property
    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """A pair of 2-d numpy arrays holding the coordinates of the box.

        Notes
        -----
        As is the default for `numpy.meshgrid`, the returned tuple is ordered
        ``(x, y)``, not ``(y, x)``, despite this being the oppposite of the
        ``(row, column)`` ordering used to index the arrays themselves.
        """
        return cast(tuple[np.ndarray, np.ndarray], tuple(np.meshgrid(self.x.arange, self.y.arange)))

    @classmethod
    def hull(cls, first: Point | Box, *args: Point | Box) -> Box:
        """Construct a new box that includes all of the given points and/or
        boxes.
        """
        x_args = [first.x]
        y_args = [first.y]
        for arg in args:
            x_args.append(arg.x)
            y_args.append(arg.y)
        return Box(x=Interval.hull(*x_args), y=Interval.hull(*y_args))

    @classmethod
    def from_size(cls, size: Extent, start: Point = Point.zero) -> Box:
        """Construct a box from its size and optional start."""
        return cls(x=Interval.from_size(size.x, start.x), y=Interval.from_size(size.y, start.y))

    def __add__(self, other: Extent) -> Box:
        return Box(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Extent) -> Box:
        return Box(x=self.x - other.x, y=self.y - other.y)

    def __contains__(self, point: Point) -> bool:
        return point.x in self.x and point.y in self.y

    def contains(self, other: Box) -> bool:
        """Test whether this box fully contains another."""
        return self.x.contains(other.x) and self.y.contains(other.y)

    def intersection(self, other: Box) -> Box | None:
        """Return a box that is contained by both ``self`` and ``other``.

        When there is no overlap between the boxes, `None` is returned.
        """
        x = self.x.intersection(other.x)
        y = self.y.intersection(other.y)
        if x is None or y is None:
            return None
        return Box.model_construct(x=x, y=y)


class BoundsFactory:
    """A factory for `Interval` and `Box` objects using array-slice syntax.

    Notes
    -----
    When indexed with a single slice, this returns an `Interval`::

        assert bounds[3:6] == Interval(start=3, stop=6)

    When indexed with a pair of slices, this returns a `Box`, with the slices
    interpreted as ``[y, x]``:

        assert (
            bounds[3:6, -1:2]
            == Box(x=Interval(start=-1, stop=2), y=Interval(start=3, stop=6)
        )
    """

    @overload
    def __getitem__(self, key: slice) -> Interval: ...

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Box: ...

    def __getitem__(self, key: slice | tuple[slice, slice]) -> Interval | Box:
        match key:
            case slice(start=start, stop=stop):
                return Interval(start=start, stop=stop)
            case (slice(start=y_start, stop=y_stop), slice(start=x_start, stop=x_stop)):
                return Box(x=Interval(start=x_start, stop=x_stop), y=Interval(start=y_start, stop=y_stop))
        raise TypeError("Unsupported slice for bounds factory.")


bounds = BoundsFactory()
