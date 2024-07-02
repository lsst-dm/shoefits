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

__all__ = ("Mask", "MaskPlane", "MaskSchema", "MaskReference")

import dataclasses
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from . import asdf_utils
from ._dtypes import NumberType
from ._geom import Box
from ._read_context import ReadContext, ReadError
from ._write_context import WriteContext, WriteError


@dataclasses.dataclass(frozen=True)
class MaskPlane:
    """Name and description of a single plane in a mask array."""

    name: str
    description: str


@dataclasses.dataclass(frozen=True)
class MaskPlaneBit:
    """The nested array index and mask value associated with a single mask
    plane.
    """

    index: int
    """Index into the last dimension of the mask array where this plane's bit
    is stored.
    """

    mask: int
    """Bitmask that select just this plane's bit from a mask array value.
    """

    @classmethod
    def compute(cls, overall_index: int, stride: int) -> MaskPlaneBit:
        """Construct from the overall index of a plane in a `MaskSchema` and
        the stride (number of bits per mask array element).
        """
        index, bit = divmod(overall_index, stride)
        return cls(index, 1 << bit)


class MaskSchema:
    """A schema for a bit-packed mask array.

    Parameters
    ----------
    planes
        Iterable of `MaskPlane` instances that define the schema.  `None`
        values may be included to reserve bits for future use.
    dtype, optional
        The numpy data type of the mask arrays that use this schema.

    Notes
    -----
    A `MaskSchema` is a collection of mask planes, which each correspond to a
    single bit in a mask array.  Mask schemas are immutable and associated with
    a particular array data type, allowing them to safely precompute the index
    and bitmask for each plane.

    `MaskSchema` indexing is by integer (the overall index of a plane in the
    schema).  The `descriptions` attribute may be index by plane name to get
    the description for that plane, and the `bitmask` method can be used to
    obtain an array that can be used to select one or more planes by name in
    a mask array that uses this schema.
    """

    def __init__(self, planes: Iterable[MaskPlane | None], dtype: npt.DTypeLike = np.uint8):
        self._planes = tuple(planes)
        self._dtype = np.dtype(dtype)
        self._descriptions = {plane.name: plane.description for plane in self._planes if plane is not None}
        stride = self._dtype.itemsize * 8
        self._mask_size = math.ceil(len(self._planes) / stride)
        self._bits: dict[str, MaskPlaneBit] = {
            plane.name: MaskPlaneBit.compute(n, stride)
            for n, plane in enumerate(self._planes)
            if plane is not None
        }

    def __iter__(self) -> Iterator[MaskPlane | None]:
        return iter(self._planes)

    def __len__(self) -> int:
        return len(self._planes)

    def __getitem__(self, i: int) -> MaskPlane | None:
        return self._planes[i]

    def __repr__(self) -> str:
        return f"MaskSchema({list(self._planes)}, dtype={self._dtype!r})"

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{name} [{bit.index}@{hex(bit.mask)}]: {self._descriptions[name]}"
                for name, bit in self._bits.items()
            ]
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MaskSchema):
            return self._planes == other._planes
        return False

    @property
    def dtype(self) -> np.dtype:
        """The numpy data type of the mask arrays that use this schema."""
        return self._dtype

    @property
    def mask_size(self) -> int:
        """The number of elements in the last dimension of any mask array that
        uses this schema.
        """
        return self._mask_size

    @property
    def descriptions(self) -> Mapping[str, str]:
        """A mapping from plane name to description."""
        return self._descriptions

    def bitmask(self, *planes: str) -> np.ndarray:
        """Return a 1-d mask array that represents the union (i.e. bitwise OR)
        of the planes with the given names.

        Parameters
        ----------
        *planes
            Mask plane names.

        Returns
        -------
        array
            A 1-d array with shape ``(mask_size,)``.
        """
        result = np.zeros(self.mask_size, dtype=self._dtype)
        for plane in planes:
            bit = self._bits[plane]
            result[bit.index] |= bit.mask
        return result


class Mask:
    """An integer array that represents a bitmask.

    Parameters
    ----------
    array_or_fill, optional
        Array or fill value for the mask.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    schema
        Schema that defines the planes and their bit assignments.
    bbox, optional
        Bounding box for the mask.  This sets the shape of all but the last
        dimension of the array.
    start, optional
        Logical coordinates of the first pixel in the array.  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape, optional
        Leading dimensions of the array.  Only needed if ``array_or_fill` is
        not an array and ``bbox`` is not provided.  Like the bbox, this does
        not include the last dimension of the array.

    Notes
    -----
    Indexing the `array` attribute of a `Mask` does not take into account its
    `start` offset, but accessing a subimage mask by indexing a `Mask` with a
    `Box` does, and the `bbox` of the subimage is set to match its location
    within the original mask.

    A mask's ``bbox`` corresponds to the leading dimensions of its backing
    `numpy.ndarray`, while the last dimension's size is always equal to the
    `~MaskSchema.mask_size` of its schema, since a schema can in general
    require multiple array elements to represent all of its planes.
    """

    def __init__(
        self,
        array_or_fill: np.ndarray | int = 0,
        /,
        *,
        schema: MaskSchema,
        bbox: Box | None = None,
        start: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ):
        if isinstance(array_or_fill, np.ndarray):
            array = np.array(array_or_fill, dtype=schema.dtype)
            if array.ndim != 3:
                raise ValueError("Mask array must be 3-d.")
            if bbox is None:
                bbox = Box.from_shape(array.shape[:-1], start=start)
            elif bbox.shape + (schema.mask_size,) != array.shape:
                raise ValueError(
                    f"Explicit bbox shape {bbox.shape} and schema of size {schema.mask_size} do not "
                    f"match array with shape {array.shape}."
                )
            if shape is not None and shape + (schema.mask_size,) != array.shape:
                raise ValueError(
                    f"Explicit shape {shape} and schema of size {schema.mask_size} do "
                    f"not match array with shape {array.shape}."
                )

        else:
            if bbox is None:
                if shape is None:
                    raise TypeError("No bbox, size, or array provided.")
                bbox = Box.from_shape(shape, start=start)
            array = np.full(bbox.shape + (schema.mask_size,), array_or_fill, dtype=schema.dtype)
        self._array = array
        self._bbox = bbox
        self._schema = schema

    @property
    def array(self) -> np.ndarray:
        """The low-level array.

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        self._array[:, :] = value

    @property
    def schema(self) -> MaskSchema:
        """Schema that defines the planes and their bit assignments."""
        return self._schema

    @property
    def bbox(self) -> Box:
        """Bounding box for the mask.  This sets the shape of all but the last
        dimension of the array.
        """
        return self._bbox

    def __getitem__(self, bbox: Box) -> Mask:
        return Mask(
            self.array[bbox.y.slice_within(self._bbox.y), bbox.x.slice_within(self._bbox.x), :],
            bbox=bbox,
            schema=self.schema,
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_ref_schema = pcs.chain_schema(
            [
                MaskReference.__pydantic_core_schema__,
                pcs.with_info_plain_validator_function(cls._from_reference),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_ref_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Mask), from_ref_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize, info_arg=True),
        )

    @classmethod
    def _from_reference(cls, reference: MaskReference, info: pydantic.ValidationInfo) -> Mask:
        schema = MaskSchema(reference.planes, dtype=reference.data.datatype.to_numpy())

        def bbox_from_shape(shape: tuple[int, ...]) -> Box:
            assert len(shape) == 3, "should be from a 3-d array"
            if shape[2] != schema.mask_size:
                raise ReadError(
                    f"Mask array shape ends with {shape[2]}, not {schema.mask_size} as expected from schema."
                )
            return Box.from_shape(shape, start=reference.start + (0,))

        slice_result: Callable[[Box], tuple[slice, ...]] | None = None
        if read_context := ReadContext.from_info(info):
            if (slice_bbox := read_context.get_parameter_bbox()) is not None:

                def slice_result(full_bbox: Box) -> tuple[slice, ...]:
                    return slice_bbox.slice_within(full_bbox) + (slice(None, None),)

        array = asdf_utils.ArraySerialization.from_model(
            reference.data,
            info,
            bbox_from_shape=bbox_from_shape,
            slice_result=slice_result,
        )

        return cls(array, start=reference.start, schema=schema)

    def _serialize(self, info: pydantic.SerializationInfo) -> MaskReference:
        if (write_context := WriteContext.from_info(info)) is None:
            raise WriteError("Cannot write mask without WriteContext in Pydantic SerializationInfo.")
        source = write_context.add_mask(self)
        data = asdf_utils.ArrayReferenceModel(
            source=source,
            shape=list(self.array.shape),
            datatype=NumberType.from_numpy(self.array.dtype).require_unsigned(),
        )
        return MaskReference(data=data, start=self.bbox.start, planes=list(self.schema))

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(MaskReference.__pydantic_core_schema__)

    def _get_array(self) -> np.ndarray:
        return self._array


class MaskReference(pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of a `Mask`."""

    data: asdf_utils.ArrayModel
    start: tuple[int, ...]
    planes: list[MaskPlane | None]
