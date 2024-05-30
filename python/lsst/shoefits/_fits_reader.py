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

__all__ = ("FitsReader", "ReadError")

import json
import math
from collections.abc import Mapping
from typing import Any, BinaryIO, Generic, TypeVar

import astropy.io.fits
import numpy as np
import pydantic

from . import asdf_utils, keywords
from ._field_info import (
    FieldInfo,
    HeaderFieldInfo,
    ImageFieldInfo,
    MappingFieldInfo,
    MaskFieldInfo,
    ModelFieldInfo,
    PolymorphicFieldInfo,
    SequenceFieldInfo,
    StructFieldInfo,
    ValueFieldInfo,
)
from ._geom import Box, Extent, Point
from ._image import Image, ImageReference
from ._mask import Mask, MaskReference, MaskSchema
from ._polymorphic import PolymorphicAdapter, PolymorphicAdapterRegistry
from ._struct import Struct
from .json_utils import JsonValue

FITS_BLOCK_SIZE = 2880


_T = TypeVar("_T", bound=Struct)


class ReadError(RuntimeError):
    pass


class FileSchemaError(ReadError):
    pass


class FitsReader(Generic[_T]):
    def __init__(
        self,
        root_type: type[_T],
        buffer: BinaryIO,
        parameters: Mapping[str, Any] | None = None,
        *,
        adapter_registry: PolymorphicAdapterRegistry,
    ):
        self._buffer = buffer
        self._root_type = root_type
        self._parameters = parameters or {}
        self._block_addresses: list[tuple[int, int]] = []
        self._ext_addresses: dict[str, int] = {}
        self._header = astropy.io.fits.Header()
        fits: astropy.io.fits.Header = astropy.io.fits.open(self._buffer)
        self._read_tree_and_addresses(fits[0])
        self._adapter_registry = adapter_registry

    def _read_tree_and_addresses(self, hdu: astropy.io.fits.PrimaryHDU) -> None:
        tree_size = hdu.header.pop(keywords.TREE_SIZE)
        tree_bytes = hdu.section[:tree_size].tobytes()
        self._tree = json.loads(tree_bytes)
        block_address = tree_size
        n = 0
        while (block_size := hdu.header.pop(keywords.ASDF_BLOCK_SIZE.format(n), None)) is not None:
            self._block_addresses.append((block_address, block_size))
            block_address += block_size
            n += 1
        n = 1
        while (ext_addr := hdu.header.pop(keywords.EXT_ADDRESS.format(n), None)) is not None:
            ext_label = hdu.header.pop(keywords.EXT_LABEL.format(n))
            self._ext_addresses[ext_label] = ext_addr
            n += 1
        self._header.update(hdu.header)
        self._header.strip()

    def read(self) -> _T:
        struct_data: dict[str, Any] = {}
        for name, field_info in self._root_type.struct_fields.items():
            struct_data[name] = self._read_dispatch(
                field_info, self._tree[name], name, frame_depth=0, target=None
            )
        return self._root_type._from_struct_data(struct_data)

    def read_component(self, component: str) -> Any:
        target = self.get_component_path(component)
        if not target:
            return self.read()
        target.reverse()
        name = target.pop()
        field_info = self._root_type.struct_fields[name]
        return self._read_dispatch(field_info, self._tree[name], name, frame_depth=0, target=target)

    def get_parameter_bbox(self, path: str, full_bbox: Box, parameters: Mapping[str, Any]) -> Box:
        return parameters.get("bbox", full_bbox)

    def get_component_path(self, component: str) -> list[str]:
        return component.split("/")

    def _read_dispatch(
        self,
        field_info: FieldInfo,
        tree: Any,
        path: str,
        frame_depth: int,
        target: list[str] | None,
    ) -> Any:
        if target is not None and not target:
            # This is the target of a single-component read. Set target to
            # None to tell the _read_* methods to stop seeking and start
            # reading.
            target = None
        match field_info:
            case ValueFieldInfo():
                return tree
            case ImageFieldInfo():
                return self._read_image(field_info, tree, path)
            case MaskFieldInfo():
                return self._read_mask(field_info, tree, path)
            case StructFieldInfo():
                return self._read_struct(field_info, tree, path, frame_depth, target)
            case MappingFieldInfo():
                return self._read_mapping(field_info, tree, path, frame_depth, target)
            case SequenceFieldInfo():
                return self._read_sequence(field_info, tree, path, frame_depth, target)
            case ModelFieldInfo():
                return self._read_model(field_info, tree, path)
            case HeaderFieldInfo():
                return self._read_header(field_info, tree, path, frame_depth)
        raise AssertionError()

    def _read_image(self, field_info: ImageFieldInfo, tree: dict[str, JsonValue], path: str) -> Image:
        image_ref = ImageReference.model_validate(tree)
        array_model, unit = image_ref.unpack()
        if unit != field_info.unit:
            raise FileSchemaError(
                f"Incorrect unit for image at {path}; expected {field_info.unit}, got {unit}."
            )
        array = self._read_array(array_model, image_ref.start, type_name=field_info.type_name, path=path)
        return Image(array, start=image_ref.start, size=Extent.from_shape(array.shape), unit=unit)

    def _read_mask(self, field_info: MaskFieldInfo, tree: dict[str, JsonValue], path: str) -> Mask:
        mask_ref = MaskReference.model_validate(tree)
        array_model = mask_ref.data
        array = self._read_array(array_model, mask_ref.start, type_name=field_info.type_name, path=path)
        schema = MaskSchema(mask_ref.planes, dtype=field_info.type_name)
        return Mask(array, start=mask_ref.start, size=Extent.from_shape(array.shape), schema=schema)

    def _read_array(
        self, array_model: asdf_utils.ArrayModel, start: Point, type_name: str, path: str
    ) -> np.ndarray:
        if array_model.datatype != type_name:
            raise FileSchemaError(
                f"Incorrect pixel type for image at {path}; expected {type_name}, got {array_model.datatype}."
            )
        match array_model:
            case asdf_utils.ArrayReferenceModel(source=str(fits_source)):
                address = self._ext_addresses[fits_source.removeprefix("fits:")]
            case asdf_utils.ArrayReferenceModel(source=int()):
                raise NotImplementedError("ASDF block reads not yet supported.")
            case asdf_utils.InlineArrayModel(data=data):
                return np.ndarray(data)
            case _:
                raise AssertionError()
        full_bbox = Box.from_size(Extent.from_shape(array_model.shape), start=start)
        bbox = self.get_parameter_bbox(path, full_bbox, self._parameters)
        dtype = np.dtype(array_model.datatype).newbyteorder("B" if array_model.byteorder == "big" else "L")
        if not full_bbox.contains(bbox):
            raise ReadError(f"Image at {path} has bbox={full_bbox}, which does not contain {bbox}.")
        start_address = address + (bbox.y.start - full_bbox.y.start) * bbox.x.size * dtype.itemsize
        if bbox.x == full_bbox.x:
            # We can read full rows because they're contiguous on disk.
            self._buffer.seek(start_address)
            array1d = np.fromfile(self._buffer, dtype=dtype, offset=0, count=math.prod(array_model.shape))
            array = array1d.reshape(*array_model.shape)
        else:
            # Read row-by-row.  We don't do any clever caching or buffering
            # because we're *hoping* that's best left to the file-like object
            # we're passed.
            array = np.zeros(array_model.shape, dtype=dtype)
            start_address += (bbox.x.start - full_bbox.x.start) * dtype.itemsize
            self._buffer.seek(start_address)
            stride = full_bbox.x.size * dtype.itemsize
            for i in range(bbox.y.size):
                array[i, :] = np.fromfile(self._buffer, dtype=dtype, offset=i * stride, count=bbox.x.size)
        if not dtype.isnative:
            array.byteswap(inplace=True)
        return array

    def _read_struct(
        self,
        field_info: StructFieldInfo,
        tree: dict[str, JsonValue],
        path: str,
        frame_depth: int,
        target: list[str] | None,
    ) -> Any:
        struct_data: dict[str, Any] = {}
        if field_info.is_frame:
            frame_depth += 1
        if target is not None:
            name = target.pop()
            return self._read_dispatch(
                field_info.cls.struct_fields[name], tree[name], f"{path}/{name}", frame_depth, target
            )
        for name, nested_field_info in field_info.cls.struct_fields.items():
            struct_data[name] = self._read_dispatch(
                nested_field_info, tree[name], f"{path}/{name}", frame_depth, target
            )
        return field_info.cls._from_struct_data(struct_data)

    def _read_mapping(
        self,
        field_info: MappingFieldInfo,
        tree: dict[str, JsonValue],
        path: str,
        frame_depth: int,
        target: list[str] | None,
    ) -> Any:
        if target is not None:
            name = target.pop()
            return self._read_dispatch(field_info.value, tree[name], f"{path}/{name}", frame_depth, target)
        load_data: dict[str, Any] = {}
        for key, value in tree.items():
            load_data[key] = self._read_dispatch(
                field_info.value, value, f"{path}/{key}", frame_depth, target
            )
        return field_info.load_factory(load_data)

    def _read_sequence(
        self,
        field_info: SequenceFieldInfo,
        tree: list[JsonValue],
        path: str,
        frame_depth: int,
        target: list[str] | None,
    ) -> Any:
        if target is not None:
            index = int(target.pop())
            return self._read_dispatch(field_info.value, tree[index], f"{path}/{index}", frame_depth, target)
        load_data: list[Any] = []
        for index, value in enumerate(tree):
            load_data[index] = self._read_dispatch(
                field_info.value, value, f"{path}/{index}", frame_depth, target
            )
        return field_info.load_factory(load_data)

    def _read_model(self, field_info: ModelFieldInfo, tree: Any, path: str) -> pydantic.BaseModel:
        return field_info.cls.model_validate(tree)

    def _read_header(
        self, field_info: HeaderFieldInfo, tree: JsonValue, path: str, frame_depth: int
    ) -> astropy.io.fits.Header:
        if frame_depth == 0:
            return self._header.copy()
        return astropy.io.fits.Header()

    def _read_polymorphic(
        self, field_info: PolymorphicFieldInfo, tree: dict[str, JsonValue], path: str, frame_depth: int
    ) -> Any:
        match tree:
            case {"tag": str(tag)}:
                pass
            case _:
                raise ReadError(f"No string 'tag' entry found for polymorphic field at {path}.")
        adapter: PolymorphicAdapter[Any, Struct] = self._adapter_registry[tag]
        if "tag" not in adapter.struct_type.struct_fields:
            del tree["tag"]
        struct = self._read_struct(field_info.as_struct(adapter.struct_type), tree, path, frame_depth, None)
        return adapter.from_struct(struct)
