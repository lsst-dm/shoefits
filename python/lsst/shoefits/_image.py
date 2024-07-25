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

__all__ = ("Image", "ImageReference")

from collections.abc import Sequence
from typing import Any, final

import astropy.io.fits
import astropy.units
import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs

from . import asdf_utils
from ._geom import Box
from ._write_context import WriteContext, WriteError


@final
class Image:
    """An array that may be augmented with units and a nonzero origin.

    Parameters
    ----------
    array_or_fill, optional
        Array or fill value for the image.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    bbox, optional
        Bounding box for the image.
    start, optional
        Logical coordinates of the first pixel in the array.  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape, optional
        Dimensions of the array.  Only needed if ``array_or_fill` is not an
        array and ``bbox`` is not provided.
    unit, optional
        Units for the image's pixel values.
    dtype, optional
        Pixel data type override.

    Notes
    -----
    Indexing the `array` attribute of an `Image` does not take into account its
    `start` offset, but accessing a subimage by indexing an `Image` with a
    `Box` does, and the `bbox` of the subimage is set to match its location
    within the original image.
    """

    def __init__(
        self,
        array_or_fill: np.ndarray | int | float = 0,
        /,
        *,
        bbox: Box | None = None,
        start: Sequence[int] | None = None,
        shape: Sequence[int] | None = None,
        unit: astropy.units.Unit | None = None,
        dtype: npt.DTypeLike | None = None,
    ):
        if isinstance(array_or_fill, np.ndarray):
            if dtype is not None:
                array = np.array(array_or_fill, dtype=dtype)
            else:
                array = array_or_fill
            if bbox is None:
                bbox = Box.from_shape(array.shape, start=start)
            elif bbox.shape != array.shape:
                raise ValueError(
                    f"Explicit bbox shape {bbox.shape} does not match array with shape {array.shape}."
                )
            if shape is not None and shape != array.shape:
                raise ValueError(f"Explicit shape {shape} does not match array with shape {array.shape}.")

        else:
            if bbox is None:
                if shape is None:
                    raise TypeError("No bbox, shape, or array provided.")
                bbox = Box.from_shape(shape, start=start)
            elif shape is not None and shape != bbox.shape:
                raise ValueError(f"Explicit shape {shape} does not match bbox shape {bbox.shape}.")
            array = np.full(bbox.shape, array_or_fill, dtype=dtype)
        self._array = array
        self._bbox = bbox
        self._unit = unit

    @property
    def array(self) -> np.ndarray:
        """The low-level array.

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray | int | float) -> None:
        self._array[...] = value

    @property
    def unit(self) -> astropy.units.Unit | None:
        """Units for the image's pixel values."""
        return self._unit

    @property
    def bbox(self) -> Box:
        """Bounding box for the image."""
        return self._bbox

    def __getitem__(self, bbox: Box) -> Image:
        return Image(self.array[bbox.slice_within(self._bbox)], bbox=bbox)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_ref_schema = pcs.chain_schema(
            [
                ImageReference.__pydantic_core_schema__,
                pcs.with_info_plain_validator_function(cls._from_reference),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_ref_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Image), from_ref_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize, info_arg=True),
        )

    @classmethod
    def _from_reference(cls, reference: ImageReference, info: pydantic.ValidationInfo) -> Image:
        array_model, unit = reference.unpack()

        def bbox_from_shape(shape: tuple[int, ...]) -> Box:
            return Box.from_shape(shape, start=reference.start)

        array = asdf_utils.ArraySerialization.from_model(array_model, info, bbox_from_shape=bbox_from_shape)
        return cls(array, start=reference.start, unit=unit)

    def _serialize(self, info: pydantic.SerializationInfo) -> ImageReference:
        if (write_context := WriteContext.from_info(info)) is None:
            raise WriteError("Cannot write image without WriteContext in Pydantic SerializationInfo.")
        header: astropy.io.fits.Header | None = None
        if write_context.get_fits_options():
            header = astropy.io.fits.Header()
            if self.unit is not None:
                header["BUNIT"] = self.unit.to_string(format="fits")
        data = write_context.add_array(
            self.array, header, start=[i.start for i in self.bbox], add_wcs_default=True
        )
        return ImageReference.pack(data, [i.start for i in self.bbox], self.unit)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        return handler(ImageReference.__pydantic_core_schema__)

class ImageReference(pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of an `Image`."""

    data: asdf_utils.QuantityModel | asdf_utils.ArrayModel
    start: list[int]

    @classmethod
    def pack(
        cls, array_model: asdf_utils.ArrayModel, start: Sequence[int], unit: asdf_utils.Unit | None
    ) -> ImageReference:
        """Construct an `ImageReference` from the components of a serialized
        image.

        Parameters
        ----------
        array_model
            Serialized form of the underlying array.
        start
            Logical coordinates of the first pixel in the array.
        unit, optional
            Units for the image's pixel values.
        """
        if unit is None:
            return cls.model_construct(data=array_model, start=list(start))
        return cls.model_construct(
            data=asdf_utils.QuantityModel.model_construct(value=array_model, unit=unit), start=list(start)
        )

    def unpack(self) -> tuple[asdf_utils.ArrayModel, asdf_utils.Unit | None]:
        """Return the components of a serialized image from this model.

        Returns
        -------
        array_model
            Serialized form of the underlying array.
        unit
            Units for the image's pixel values.

        Notes
        -----
        The ``start`` attribute is not included in the results because it is
        directly accessible, rather than possibly nested under `data` as with
        the other attributes.
        """
        if isinstance(self.data, asdf_utils.QuantityModel):
            if not isinstance(self.data.value, asdf_utils.ArrayReferenceModel | asdf_utils.InlineArrayModel):
                raise ValueError("Expected array quantity, not scalar.")
            return self.data.value, self.data.unit
        return self.data, None
