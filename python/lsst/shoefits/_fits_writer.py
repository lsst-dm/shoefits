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

__all__ = ("FitsWriter", "WriteError")


import dataclasses
import json
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from io import BytesIO, TextIOWrapper
from typing import Any, BinaryIO, Literal, TypedDict, cast

import astropy.io.fits
import astropy.units
import numpy as np
import pydantic

from . import asdf_utils
from ._compression import FitsCompression
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
from ._frame import Frame
from ._geom import Point
from ._image import Image, ImageReference
from ._mask import Mask, MaskReference, MaskSchema
from ._polymorphic import PolymorphicAdapterRegistry
from ._struct import Struct
from .json_utils import JsonValue

FORMAT_VERSION = (0, 0, 1)


class WriteError(RuntimeError):
    pass


class AddressedTreeData(TypedDict, total=False):
    address: int


class SkipNode(BaseException):
    pass


@contextmanager
def handle_skips() -> Iterator[None]:
    try:
        yield
    except SkipNode:
        pass


@dataclasses.dataclass
class FitsExtensionLabel:
    extname: str
    extver: int | None
    extlevel: int

    @property
    def asdf_source(self) -> str:
        result = f"fits:{self.extname}"
        if self.extver is not None:
            result = f"{result},{self.extver}"
        return result

    def update_header(self, header: astropy.io.fits.Header) -> None:
        header["EXTNAME"] = self.extname
        if self.extver is not None:
            header["EXTVER"] = self.extver
        header["EXTLEVEL"] = self.extlevel


@dataclasses.dataclass
class FitsExtensionWriter:
    path: Path
    parent_header: astropy.io.fits.Header | None
    extension_only_header: astropy.io.fits.Header
    array: np.ndarray
    addressed: AddressedTreeData
    compression: FitsCompression | None = None


@dataclasses.dataclass(frozen=True)
class Path:
    to_frame: tuple[str | int, ...] = ()
    from_frame: tuple[str | int, ...] = ()
    extlevel: int = 1

    def push(self, term: str | int) -> Path:
        return Path(self.to_frame, self.from_frame + (term,), extlevel=self.extlevel)

    def reset(self) -> Path:
        return Path(self.to_frame + self.from_frame, (), extlevel=self.extlevel + 1)

    @property
    def default_header_key(self) -> str:
        return "-".join(str(term).upper() for term in self.from_frame)

    @property
    def default_extension_label(self) -> FitsExtensionLabel:
        full_path = list(self.to_frame + self.from_frame)
        extver = None
        for index in reversed(range(len(full_path))):
            if isinstance(term := full_path[index], int):
                extver = term
                del full_path[index]
                break
        extname = "/".join(str(t) for t in full_path)
        return FitsExtensionLabel(extname=extname, extver=extver, extlevel=self.extlevel)


class FitsWriter:
    def __init__(self, struct: Struct, adapter_registry: PolymorphicAdapterRegistry):
        self.primary_hdu = astropy.io.fits.PrimaryHDU()
        self.primary_hdu.header.set(
            "SHOEFITS", ".".join(str(v) for v in FORMAT_VERSION), "SHOEFITS format version."
        )
        self.primary_hdu.header.set("TREEADDR", 0, "Address of JSON tree (bytes from file start).")
        self.primary_hdu.header.set("TREESIZE", 0, "Size of JSON tree (bytes).")
        self.extension_writers: list[FitsExtensionWriter] = []
        self.block_writer = asdf_utils.BlockWriter()
        path = Path()
        if is_frame := isinstance(struct, Frame):
            path.reset()
        self.primary_header = astropy.io.fits.Header()
        self.hdu_addressed: list[AddressedTreeData] = [{}]
        # Squash astropy warnings about needing HIERARCH, since the only way to
        # get it to do HIERARCH only when needed is to let it warn.
        with warnings.catch_warnings(category=astropy.io.fits.verify.VerifyWarning, action="ignore"):
            self.tree = self._walk_struct(struct, path, header=None, is_frame=is_frame)
        for n, block_size in enumerate(self.block_writer.sizes()):
            self.primary_hdu.header.set(f"BLK{n:05d}", block_size, f"Size of ASDF block {n} (bytes).")
        self.primary_hdu.header.update(self.primary_header)
        self._adapter_registry = adapter_registry

    def write(self, buffer: BinaryIO) -> None:
        address = 0
        hdu_list = astropy.io.fits.HDUList([self.primary_hdu])
        # There's no method to get the size of the header without stringifying
        # it, so that's what we do (here and later).  The primary header isn't
        # actually complete - we need to fill in the TREEADDR and TREESIZE
        # headers once we have finalized the tree.  But that shouldn't affect
        # the header size, because we've already put those cards in with
        # padding zeros.
        address += len(self.primary_hdu.header.tostring())
        for writer in self.extension_writers:
            if writer.parent_header is None:
                full_header = astropy.io.fits.Header()
            else:
                full_header = writer.parent_header.copy()
            full_header.set("INHERIT", True)
            full_header.update(writer.extension_only_header)
            if compression := self.get_compression(writer.path, writer.compression, writer.array):
                tile_shape = compression.tile_size.shape + writer.array.shape[2:]
                hdu = astropy.io.fits.CompImageHDU(
                    writer.array,
                    header=full_header,
                    compression_type=compression.algorithm.value,
                    tile_shape=tile_shape,
                )
                raise NotImplementedError(
                    "TODO: this triggers some internal error in astropy when we try "
                    "to write the HDUList later.  Not sure what's unusual here."
                )
            else:
                hdu = astropy.io.fits.ImageHDU(writer.array, full_header)
            writer.addressed["address"] = address + len(hdu.header.tostring())
            address += hdu.filebytes()
            hdu_list.append(hdu)
        tree_fits_header = astropy.io.fits.Header()
        tree_fits_header["XTENSION"] = "IMAGE"
        tree_fits_header["BITPIX"] = 8
        tree_fits_header["NAXIS"] = 1
        tree_fits_header["NAXIS1"] = 0
        tree_fits_header["EXTNAME"] = "JSONTREE"
        self.primary_hdu.header["TREEADDR"] = address + len(tree_fits_header.tostring())
        asdf_buffer = BytesIO()
        tree_buffer = TextIOWrapper(asdf_buffer, write_through=True)
        json.dump(self.tree, tree_buffer, ensure_ascii=False)
        tree_size = asdf_buffer.tell()
        self.primary_hdu.header["TREESIZE"] = tree_size
        self.block_writer.write(asdf_buffer)
        asdf_array = np.frombuffer(asdf_buffer.getbuffer(), dtype=np.uint8)
        hdu_list.append(astropy.io.fits.ImageHDU(asdf_array, header=tree_fits_header))
        hdu_list.writeto(buffer)  # TODO: make space for, then add checksums

    def get_header_key(self, path: Path, from_field: str | bool) -> str | None:
        if from_field:
            if from_field is True:
                return path.default_header_key
            else:
                return from_field
        return None

    def get_extension_label(self, path: Path, from_field: str | bool) -> FitsExtensionLabel | None:
        if from_field:
            if from_field is True:
                return path.default_extension_label
            else:
                return FitsExtensionLabel(extname=from_field, extver=None, extlevel=path.extlevel)
        return None

    def get_compression(
        self, path: Path, from_field: FitsCompression | None, array: np.ndarray
    ) -> FitsCompression | None:
        return from_field

    def add_mask_schema_header(
        self,
        header: astropy.io.fits.Header,
        schema: MaskSchema,
        path: Path,
        field_style: Literal["afw"] | None,
    ) -> None:
        if field_style == "afw":
            for mask_plane_index, mask_plane in enumerate(schema):
                if mask_plane is not None:
                    header.set(f"MP_{mask_plane.name.upper()}", mask_plane_index, mask_plane.description)

    def _walk_dispatch(
        self,
        value: Any,
        field_info: FieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        match field_info:
            case ValueFieldInfo():
                return self._walk_value(value, field_info, path, header)
            case ImageFieldInfo():
                return self._walk_image(value, field_info, path, header)
            case MaskFieldInfo():
                return self._walk_mask(value, field_info, path, header)
            case StructFieldInfo():
                if field_info.is_frame:
                    path = path.reset()
                return self._walk_struct(
                    value,
                    path=path,
                    header=header,
                    is_frame=field_info.is_frame,
                )
            case MappingFieldInfo():
                return self._walk_mapping(value, field_info, path, header)
            case SequenceFieldInfo():
                return self._walk_sequence(value, field_info, path, header)
            case ModelFieldInfo():
                return self._walk_model(value, field_info, path, header)
            case HeaderFieldInfo():
                return self._walk_header(value, field_info, path, header)
            case PolymorphicFieldInfo():
                return self._walk_polymorphic(value, field_info, path, header)
        raise AssertionError()

    def _walk_value(
        self,
        value: int | str | float | None,
        field_info: ValueFieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        if header is None:
            header = self.primary_header
        if header_key := self.get_header_key(path, field_info.fits_header):
            header.set(header_key, value, field_info.description)
        return value

    def _walk_image(
        self, image: Image, field_info: ImageFieldInfo, path: Path, header: astropy.io.fits.Header | None
    ) -> JsonValue:
        source: int | str
        if label := self.get_extension_label(path, field_info.fits_image_extension):
            extension_only_header = astropy.io.fits.Header()
            if image.unit is not None:
                extension_only_header["BUNIT"] = image.unit.to_string(format="fits")
            label.update_header(extension_only_header)
            self._add_array_start_wcs(image.bbox.start, extension_only_header)
            result = ImageReference.from_image_and_source(image, label.asdf_source).model_dump()
            self.extension_writers.append(
                FitsExtensionWriter(
                    path=path,
                    array=image.array,
                    parent_header=header,
                    extension_only_header=extension_only_header,
                    addressed=cast(AddressedTreeData, result),
                )
            )
            return result
        else:
            source = self.block_writer.add_array(image.array)
            return ImageReference.from_image_and_source(image, source).model_dump()

    def _walk_mask(
        self, mask: Mask, field_info: MaskFieldInfo, path: Path, header: astropy.io.fits.Header | None
    ) -> JsonValue:
        source: int | str
        if label := self.get_extension_label(path, field_info.fits_image_extension):
            extension_only_header = astropy.io.fits.Header()
            label.update_header(extension_only_header)
            self._add_array_start_wcs(mask.bbox.start, extension_only_header)
            self.add_mask_schema_header(
                extension_only_header, mask.schema, path, field_info.fits_plane_header_style
            )
            result = MaskReference.from_mask_and_source(mask, label.asdf_source).model_dump()
            self.extension_writers.append(
                FitsExtensionWriter(
                    path=path,
                    array=mask.array,
                    parent_header=header,
                    extension_only_header=extension_only_header,
                    addressed=cast(AddressedTreeData, result),
                    compression=field_info.fits_compression,
                )
            )
            return result
        else:
            source = self.block_writer.add_array(mask.array)
            return MaskReference.from_mask_and_source(mask, source).model_dump()

    def _walk_struct(
        self,
        struct: Struct,
        path: Path,
        header: astropy.io.fits.Header | None,
        is_frame: bool,
    ) -> dict[str, JsonValue]:
        if is_frame:
            if header is None:
                header = astropy.io.fits.Header()
            else:
                header = header.copy()
        result_data: dict[str, JsonValue] = {}
        for name, field_info in struct.struct_fields.items():
            with handle_skips():
                result_data[name] = self._walk_dispatch(
                    getattr(struct, name), field_info, path.push(name), header
                )
        return result_data

    def _walk_mapping(
        self,
        mapping: Mapping[str, Any],
        field_info: MappingFieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        result: dict[str, JsonValue] = {}
        for name, value in mapping.items():
            with handle_skips():
                result[name] = self._walk_dispatch(value, field_info.value, path.push(name), header)
        return result

    def _walk_sequence(
        self,
        sequence: Sequence[Any],
        field_info: SequenceFieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        result: list[JsonValue] = []
        for index, value in enumerate(sequence):
            with handle_skips():
                result.append(self._walk_dispatch(value, field_info.value, path.push(index), header))
        return result

    def _walk_model(
        self,
        model: pydantic.BaseModel,
        field_info: ModelFieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        context = {"block_writer": self.block_writer}
        return model.model_dump(context=context)

    def _walk_header(
        self,
        value: astropy.io.fits.Header,
        field_info: HeaderFieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        if header is not None:
            if value:
                warnings.warn(
                    f"Header field at {path} is nested within a frame other than the root of the tree "
                    "being written to disk, and hence cannot be populated read. Clear the header field "
                    "writing to avoid this warning (there is no warning on read)."
                )
                header.update(value)
            raise SkipNode()
        self.primary_header.update(value)
        raise SkipNode()

    def _walk_polymorphic(
        self,
        obj: Any,
        field_info: PolymorphicFieldInfo,
        path: Path,
        header: astropy.io.fits.Header | None,
    ) -> JsonValue:
        tag = field_info.get_tag(obj)
        adapter = self._adapter_registry[tag]
        struct = adapter.to_struct(obj)
        if is_frame := isinstance(struct, Frame):
            path = path.reset()
        data = self._walk_struct(struct, path, header, is_frame=is_frame)
        if data.setdefault("tag", tag) != tag:
            raise WriteError(
                f"Serialized form of {path} already has tag={data['tag']!r}, "
                f"which is inconsistent with tag={tag!r} from the get_tag callback."
            )
        return data

    def _add_array_start_wcs(self, start: Point, header: astropy.io.fits.Header, wcs_name: str = "A") -> None:
        header.set(f"CTYPE1{wcs_name}", "LINEAR", "Type of projection")
        header.set(f"CTYPE2{wcs_name}", "LINEAR", "Type of projection")
        header.set(f"CRPIX1{wcs_name}", 1.0, "Column Pixel Coordinate of Reference")
        header.set(f"CRPIX2{wcs_name}", 1.0, "Row Pixel Coordinate of Reference")
        header.set(f"CRVAL1{wcs_name}", start.x, "Column pixel of Reference Pixel")
        header.set(f"CRVAL2{wcs_name}", start.y, "Row pixel of Reference Pixel")
        header.set(f"CUNIT1{wcs_name}", "PIXEL", "Column unit")
        header.set(f"CUNIT2{wcs_name}", "PIXEL", "Row unit")
