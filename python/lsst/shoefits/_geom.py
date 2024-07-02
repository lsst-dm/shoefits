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

__all__ = (
    "Interval",
    "Box",
)

from typing import ClassVar, final

import numpy as np
import pydantic


@final
class Interval(pydantic.BaseModel):
    """A 1-d integer interval with positive size.

    Notes
    -----
    Adding or subtracting an `int` from an interval returns a shifted interval.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    factory: ClassVar[IntervalSliceFactory]

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
        """Return the `slice` that corresponds to the values in this interval
        when the items of the container being sliced correspond to ``other``.

        This assumes ``other.contains(self)``.
        """
        return slice(self.start - other.start, self.stop - other.start)


class IntervalSliceFactory:
    """A factory for `Interval` objects using array-slice syntax.

    Notes
    -----
    When indexed with a single slice, this returns an `Interval`::

        assert Interval.factory[3:6] == Interval(start=3, stop=6)

    """

    def __getitem__(self, s: slice) -> Interval:
        if s.step is not None and s.step != 1:
            raise ValueError(f"Slice {s} has non-unit step.")
        return Interval(start=s.start, stop=s.stop)


Interval.factory = IntervalSliceFactory()


class Box(pydantic.RootModel[tuple[Interval, ...]]):
    """An axis-aligned [hyper]rectangular region."""

    model_config = pydantic.ConfigDict(frozen=True)

    factory: ClassVar[BoxSliceFactory]

    @classmethod
    def from_shape(cls, shape: tuple[int, ...], start: tuple[int, ...] | None = None) -> Box:
        """Construct a box from its shape and optional start."""
        if start is None:
            start = (0,) * len(shape)
        return Box(
            root=tuple(
                [Interval.from_size(size, start=i_start) for size, i_start in zip(shape, start, strict=True)]
            )
        )

    @property
    def start(self) -> tuple[int, ...]:
        """Inclusive minimum point in the box."""
        return tuple([i.start for i in self.root])

    @property
    def stop(self) -> tuple[int, ...]:
        """One past the maximum point in the box."""
        return tuple([i.stop for i in self.root])

    @property
    def min(self) -> tuple[int, ...]:
        """Inclusive minimum point in the box."""
        return tuple([i.min for i in self.root])

    @property
    def max(self) -> tuple[int, ...]:
        """Inclusive maximum point in the box."""
        return tuple([i.max for i in self.root])

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple holding the sizes of the intervals in all dimension."""
        return tuple([i.size for i in self.root])

    @property
    def x(self) -> Interval:
        """Shortcut for the last dimension's interval."""
        return self.root[-1]

    @property
    def y(self) -> Interval:
        """Shortcut for the second-to-last dimension's interval."""
        return self.root[-2]

    @property
    def z(self) -> Interval:
        """Shortcut for the third-to-last dimension's interval."""
        return self.root[-3]

    def __str__(self) -> str:
        return f'[{",".join([str(i) for i in self.root])}]'

    def __contains__(self, x: tuple[int, ...]) -> bool:
        return all(b in a for a, b in zip(self.root, x, strict=True))

    def contains(self, other: Box) -> bool:
        """Test whether this box fully contains another."""
        return all(a.contains(b) for a, b in zip(self.root, other.root, strict=True))

    def intersection(self, other: Box) -> Box | None:
        """Return a box that is contained by both ``self`` and ``other``.

        When there is no overlap between the box, `None` is returned.
        """
        root = []
        for a, b in zip(self.root, other.root, strict=True):
            if (r := a.intersection(b)) is None:
                return None
            root.append(r)
        return Box(root=tuple(root))

    def slice_within(self, other: Box) -> tuple[slice, ...]:
        """Return a tuple of `slice` objects that corresponds to the values of
        this box when the items of the container being sliced correspond to
        ``other``.

        This assumes ``other.contains(self)``.
        """
        return tuple([a.slice_within(b) for a, b in zip(self.root, other.root, strict=True)])


class BoxSliceFactory:
    """A factory for `Box` objects using array-slice syntax.

    Notes
    -----
    When indexed with one or more slices, this returns a `Box`:

        assert (
            bounds[3:6, -1:2]
            == Box(x=Interval(start=-1, stop=2), y=Interval(start=3, stop=6)
        )
    """

    def __getitem__(self, key: slice | tuple[slice, ...]) -> Box:
        match key:
            case slice():
                return Box(root=(Interval.factory[key],))
            case tuple():
                return Box(root=tuple([Interval.factory[s] for s in key]))
            case _:
                raise TypeError("Expected slice or tuple of slices.")


Box.factory = BoxSliceFactory()
