from __future__ import annotations

__all__ = ("Point", "Extent", "Interval", "Box", "bounds")

from typing import Any, ClassVar, cast, final, overload

import numpy as np
import pydantic


class BaseGeometry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)


@final
class Point(BaseGeometry):
    x: int
    y: int

    zero: ClassVar[Point]

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
        return Extent(x=self.x, y=self.y)


Point.zero = Point(x=0, y=0)


@final
class Extent(BaseGeometry):
    x: int = 0
    y: int = 0

    zero: ClassVar[Extent]

    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> Extent:
        return cls(x=shape[1], y=shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y, self.x)

    def __add__(self, other: Extent) -> Extent:
        return Extent(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Extent) -> Extent:
        return Extent(x=self.x - other.x, y=self.y - other.y)


Extent.zero = Extent(x=0, y=0)


@final
class Interval(BaseGeometry):
    start: int
    stop: int

    @property
    def min(self) -> int:
        return self.start

    @property
    def max(self) -> int:
        return self.stop - 1

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def range(self) -> range:
        return range(self.start, self.stop)

    @property
    def arange(self) -> np.ndarray:
        return np.arange(self.start, self.stop)

    def __str__(self) -> str:
        return f"{self.start}:{self.stop}"

    @classmethod
    def hull(cls, first: int | Interval, *args: int | Interval) -> Interval:
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
        return cls(start=start, stop=start + size)

    def __add__(self, other: int) -> Interval:
        return Interval(start=self.start + other, stop=self.stop + other)

    def __sub__(self, other: int) -> Interval:
        return Interval(start=self.start - other, stop=self.stop - other)

    @pydantic.model_validator(mode="after")
    def _validate(self) -> Interval:
        if self.start >= self.stop:
            raise ValueError("Intervals must have nonnegative size.")
        return self


@final
class Box(BaseGeometry):
    x: Interval
    y: Interval

    @property
    def min(self) -> Point:
        return Point(x=self.x.min, y=self.y.min)

    @property
    def max(self) -> Point:
        return Point(x=self.x.max, y=self.y.max)

    @property
    def start(self) -> Point:
        return Point(x=self.x.start, y=self.y.start)

    @property
    def stop(self) -> Point:
        return Point(x=self.x.stop, y=self.y.stop)

    @property
    def size(self) -> Extent:
        return Extent(x=self.x.size, y=self.y.size)

    @property
    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return cast(tuple[np.ndarray, np.ndarray], tuple(np.meshgrid(self.x.arange, self.y.arange)))

    @classmethod
    def hull(cls, *args: Point | Box) -> Box | None:
        if not args:
            return None
        x_args = []
        y_args = []
        for arg in args:
            x_args.append(arg.x)
            y_args.append(arg.y)
        return Box(x=Interval.hull(*x_args), y=Interval.hull(*y_args))

    @classmethod
    def from_size(cls, size: Extent, start: Point = Point.zero) -> Box:
        return cls(x=Interval.from_size(size.x, start.x), y=Interval.from_size(size.y, start.y))

    def __add__(self, other: Extent) -> Box:
        return Box(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Extent) -> Box:
        return Box(x=self.x - other.x, y=self.y - other.y)


class BoundsFactory:
    def __getitem__(self, key: slice | tuple[slice, slice]) -> Interval | Box:
        match key:
            case slice(start=start, stop=stop):
                return Interval(start=start, stop=stop)
            case (slice(start=y_start, stop=y_stop), slice(start=x_start, stop=x_stop)):
                return Box(x=Interval(start=x_start, stop=x_stop), y=Interval(start=y_start, stop=y_stop))
        raise TypeError("Unsupported slice for bounds factory.")


bounds = BoundsFactory()
