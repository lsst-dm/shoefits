from __future__ import annotations

__all__ = ("FitsWriter",)


import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, BinaryIO, NotRequired, TypedDict, cast

import astropy.io.fits
import astropy.units
import pydantic
import yaml

from ._asdf import BlockWriter
from ._dtypes import BUILTIN_TYPES
from ._field_info import (
    FieldInfo,
    HeaderFieldInfo,
    ImageFieldInfo,
    MappingFieldInfo,
    MaskFieldInfo,
    ModelFieldInfo,
    SequenceFieldInfo,
    StructFieldInfo,
    ValueFieldInfo,
)
from ._frame import Frame
from ._geom import Point
from ._image import Image, ImageReference
from ._mask import Mask, MaskReference
from ._struct import Struct
from ._yaml import YamlValue

FORMAT_VERSION = (0, 0, 1)


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


class SerializationContext(TypedDict):
    yaml: bool
    block_writer: NotRequired[BlockWriter]
    addressed: NotRequired[AddressedTreeData]


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
    def __init__(self, struct: Struct):
        self.primary_hdu = astropy.io.fits.PrimaryHDU()
        self.primary_hdu.header.set(
            "SHOEFITS", ".".join(str(v) for v in FORMAT_VERSION), "SHOEFITS format version."
        )
        self.primary_hdu.header.set("TREEADDR", 0, "Address of YAML tree (bytes from file start).")
        self.primary_hdu.header.set("TREESIZE", 0, "Size of YAML tree (bytes).")
        self.hdu_list = astropy.io.fits.HDUList([self.primary_hdu])
        self.block_writer: BlockWriter = BlockWriter()
        path = Path()
        if is_frame := isinstance(struct, Frame):
            path.reset()
        primary_header = astropy.io.fits.Header()
        self.hdu_addressed: list[AddressedTreeData] = [{}]
        self.tree = self._walk_struct(struct, path, header=primary_header, is_frame=is_frame)
        self.primary_hdu.header.update(primary_header)

    def write(self, buffer: BinaryIO) -> None:
        buffer_address = buffer.tell()
        address = buffer_address
        self.hdu_list.writeto(buffer, overwrite=True, checksum=True)
        for hdu, addressed in zip(self.hdu_list, self.hdu_addressed):
            # hdu.fileinfo() doesn't seem to work (at least on some file-like
            # objects) and there's no method to get the size of the header
            # without stringifying it, so that's what we do.
            addressed["address"] = address + len(hdu.header.tostring())
            address += hdu.filebytes()
        tree_header_address = buffer.tell()
        tree_fits_header = astropy.io.fits.Header()
        tree_fits_header["XTENSION"] = "IMAGE"
        tree_fits_header["BITPIX"] = 8
        tree_fits_header["NAXIS"] = 1
        tree_fits_header["NAXIS1"] = 0
        tree_fits_header["EXTNAME"] = "ASDF"
        tree_fits_header.tofile(buffer)
        tree_data_address = buffer.tell()
        buffer.writelines(
            [
                b"#ASDF 1.0.0\n",
                b"%YAML 1.1\n",
                b"%TAG ! tag:stsci.edu:asdf/\n",
                b"--- !core/asdf-1.0.0\n",
            ]
        )
        yaml.dump(self.tree, buffer, explicit_end=True, encoding="utf-8")
        tree_size = buffer.tell() - tree_data_address
        self.block_writer.write(buffer)
        asdf_data_size = buffer.tell() - tree_data_address
        padding = 2880 - remainder if (remainder := asdf_data_size % 2880) else 0
        buffer.write(b"\0" * padding)
        # Rewrite the ASDF extension's NAXIS to reflect the tree + blocks size.
        buffer.seek(tree_header_address)
        tree_fits_header.set("NAXIS1", asdf_data_size)
        tree_fits_header.tofile(buffer)
        # Rewrite the primary FITS header with the tree's address and size.
        buffer.seek(buffer_address)
        self.primary_hdu.header["TREEADDR"] = tree_data_address - buffer_address
        self.primary_hdu.header["TREESIZE"] = tree_size
        self.primary_hdu.header.tofile(buffer)

    def _walk_dispatch(
        self,
        value: Any,
        field_info: FieldInfo,
        path: Path,
        header: astropy.io.fits.Header,
    ) -> YamlValue:
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
        raise AssertionError()

    def _walk_value(
        self, value: object, field_info: ValueFieldInfo, path: Path, header: astropy.io.fits.Header
    ) -> YamlValue:
        value = cast(int | str | float | None, BUILTIN_TYPES[field_info.dtype](value))
        if field_info.fits_header:
            if field_info.fits_header is True:
                header_key = path.default_header_key
            else:
                header_key = field_info.fits_header
            header.set(header_key, value, field_info.description)
        return value

    def _walk_image(
        self, image: Image, field_info: ImageFieldInfo, path: Path, header: astropy.io.fits.Header
    ) -> YamlValue:
        source: int | str
        if field_info.fits_image_extension:
            if field_info.fits_image_extension is True:
                label = path.default_extension_label
            else:
                label = FitsExtensionLabel(
                    extname=field_info.fits_image_extension, extver=None, extlevel=path.extlevel
                )
            header = header.copy()
            if image.unit is not None:
                header["BUNIT"] = image.unit.to_string(format="fits")
            label.update_header(header)
            self._add_array_start_wcs(image.bbox.start, header)
            hdu = astropy.io.fits.ImageHDU(image.array, header=header)
            self.hdu_list.append(hdu)
            source = label.asdf_source
        else:
            source = self.block_writer.add_array(image.array)
        with self._serialization_context(fits=bool(field_info.fits_image_extension)) as context:
            return ImageReference.from_image_and_source(image, source).model_dump(context=context)

    def _walk_mask(
        self, mask: Mask, field_info: MaskFieldInfo, path: Path, header: astropy.io.fits.Header
    ) -> YamlValue:
        source: int | str
        if field_info.fits_image_extension:
            if field_info.fits_image_extension is True:
                label = path.default_extension_label
            else:
                label = FitsExtensionLabel(
                    extname=field_info.fits_image_extension, extver=None, extlevel=path.extlevel
                )
            header = header.copy()
            label.update_header(header)
            self._add_array_start_wcs(mask.bbox.start, header)
            if field_info.fits_plane_header_style == "afw":
                for mask_plane_index, mask_plane in enumerate(mask.schema):
                    if mask_plane is not None:
                        header.set(f"MP_{mask_plane.name.upper()}", mask_plane_index, mask_plane.description)
            hdu = astropy.io.fits.ImageHDU(mask.array, header=header)
            self.hdu_list.append(hdu)
            source = label.asdf_source
        else:
            source = self.block_writer.add_array(mask.array)
        with self._serialization_context(fits=bool(field_info.fits_image_extension)) as context:
            return MaskReference.from_mask_and_source(mask, source).model_dump(context=context)

    def _walk_struct(
        self,
        struct: Struct,
        path: Path,
        header: astropy.io.fits.Header,
        is_frame: bool,
    ) -> YamlValue:
        if is_frame:
            header = header.copy()
        result_data: dict[str, YamlValue] = {}
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
        header: astropy.io.fits.Header,
    ) -> YamlValue:
        result: dict[str, YamlValue] = {}
        for name, value in mapping.items():
            with handle_skips():
                result[name] = self._walk_dispatch(value, field_info.value, path.push(name), header)
        return result

    def _walk_sequence(
        self,
        sequence: Sequence[Any],
        field_info: SequenceFieldInfo,
        path: Path,
        header: astropy.io.fits.Header,
    ) -> YamlValue:
        result: list[YamlValue] = []
        for index, value in enumerate(sequence):
            with handle_skips():
                result.append(self._walk_dispatch(value, field_info.value, path.push(index), header))
        return result

    def _walk_model(
        self,
        model: pydantic.BaseModel,
        field_info: ModelFieldInfo,
        path: Path,
        header: astropy.io.fits.Header,
    ) -> YamlValue:
        context = {"yaml": True, "block_writer": self.block_writer}
        return model.model_dump(context=context)

    def _walk_header(
        self,
        value: astropy.io.fits.Header,
        field_info: HeaderFieldInfo,
        path: Path,
        header: astropy.io.fits.Header,
    ) -> YamlValue:
        header.update(value)
        raise SkipNode()

    def _add_array_start_wcs(self, start: Point, header: astropy.io.fits.Header, wcs_name: str = "A") -> None:
        header.set(f"CTYPE1{wcs_name}", "LINEAR", "Type of projection")
        header.set(f"CTYPE2{wcs_name}", "LINEAR", "Type of projection")
        header.set(f"CRPIX1{wcs_name}", 1.0, "Column Pixel Coordinate of Reference")
        header.set(f"CRPIX2{wcs_name}", 1.0, "Row Pixel Coordinate of Reference")
        header.set(f"CRVAL1{wcs_name}", start.x, "Column pixel of Reference Pixel")
        header.set(f"CRVAL2{wcs_name}", start.y, "Row pixel of Reference Pixel")
        header.set(f"CUNIT1{wcs_name}", "PIXEL", "Column unit")
        header.set(f"CUNIT2{wcs_name}", "PIXEL", "Row unit")

    @contextmanager
    def _serialization_context(self, fits: bool) -> Iterator[dict[str, Any]]:
        context: SerializationContext = {"yaml": True}
        if not fits:
            context["block_writer"] = self.block_writer
        # Cast is necessary because dict is mutable and hence its value type
        # is invariant, not covariant.
        yield cast(dict[str, Any], context)
        if (addressed := context.pop("addressed", None)) is not None:
            if fits:
                self.hdu_addressed.append(addressed)
