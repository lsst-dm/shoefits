from __future__ import annotations

__all__ = (
    "FitsHeaderSchema",
    "FitsValueHeaderSchema",
    "FitsImageHeaderSchema",
    "FitsColumnSchema",
    "FitsValueColumnSchema",
    "FitsImageColumnSchema",
    "FitsLabelColumnSchema",
    "FitsDataSchema",
    "FitsBinaryTableSchema",
    "FitsGridDataSchema",
    "FitsImageDataSchema",
    "FitsExtensionLabelSchema",
    "FitsExtensionSchema",
)


import dataclasses
from abc import ABC
from collections.abc import Sequence

from ._frame import (
    FieldInfo,
    Frame,
    FrameFieldInfo,
    HeaderFieldInfo,
    ImageFieldInfo,
    MappingFieldInfo,
    MaskFieldInfo,
    ValueFieldInfo,
)
from ._schema_path import Placeholders, SchemaPath, SchemaPathName


class FitsHeaderSchema(ABC):
    pass


@dataclasses.dataclass
class FitsValueHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS header cards exported from a simple `str`,
    `int`, or `float` value in the YAML tree.
    """

    key: SchemaPathName
    field_info: ValueFieldInfo
    path_from_frame: SchemaPath
    """Path of the value field, relative to `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsOpaqueHeaderSchema(FitsHeaderSchema):
    """Schema definition for a bundle of FITS opaque header cards that do not
    appear in the YAML tree at all, and are typically just propagated from
    some externally-written FITS file.
    """

    field_info: HeaderFieldInfo

    path_from_frame: SchemaPath
    """Path of the opaque-header field, relative to
    `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsImageHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS headers exported by all `Image` objects."""

    field_info: ImageFieldInfo
    path_from_frame: SchemaPath
    """Path of the image field, relative to `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsMaskHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS headers exported by all `Mask` objects."""

    field_info: MaskFieldInfo
    path_from_frame: SchemaPath
    """Path of the mask field, relative to `FitsExtensionSchema.frame_path`.
    """


class FitsColumnSchema(ABC):
    pass


@dataclasses.dataclass
class FitsLabelColumnSchema(FitsColumnSchema):
    name: str
    value: SchemaPathName


@dataclasses.dataclass
class FitsImageColumnSchema(FitsColumnSchema):
    name: str
    field_info: ImageFieldInfo
    path: SchemaPath
    """Path of the image field, relative to `FitsBinaryTableSchema.row_path`.

    This path is guaranteed not to hold any mapping or sequence placeholders.
    """


@dataclasses.dataclass
class FitsValueColumnSchema(FitsColumnSchema):
    name: str
    field_info: ValueFieldInfo
    path: SchemaPath
    """Path of the value field, relative to `FitsBinaryTableSchema.row_path`.

    This path is guaranteed not to hold any mapping or sequence placeholders.
    """


class FitsDataSchema(ABC):
    pass


@dataclasses.dataclass
class FitsBinaryTableSchema(FitsDataSchema):
    table_path: SchemaPath
    """Path of the outermost container whose whose items make up the rows of
    this table, relative to `FitsExtensionSchema.frame_path`.
    """

    row_path: SchemaPath
    """Path that corresponds to a row of this table, relative to `table_path`.

    This path holds only mapping and sequence placeholders, all of which are
    flattened out (depth first) to form the rows of the table.
    """

    columns: list[FitsColumnSchema] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FitsGridDataSchema(FitsDataSchema):
    grid_path: SchemaPath
    """Path of the outermost container whose items make up the cells of the
    grid, relative to `FitsExtensionSchema.frame_path`.
    """

    cell: FitsDataSchema

    # TODO: need to represent how container keys/indexes map to grid cells,
    # as well as how to get the bboxes of cells.


@dataclasses.dataclass
class FitsImageDataSchema(FitsDataSchema):
    field_info: ImageFieldInfo
    path_from_frame: SchemaPath
    """Path of the `Image` that maps to this FITS extension, relative to
    `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsMaskDataSchema(FitsDataSchema):
    field_info: MaskFieldInfo
    path_from_frame: SchemaPath
    """Path of the `Mask` that maps to this FITS extension, relative to
    `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsExtensionLabelSchema:
    """Schema for the identifier header columns in a FITS extension."""

    extname: SchemaPathName
    extver: SchemaPath | None
    extlevel: int


@dataclasses.dataclass
class FitsExtensionSchema:
    label: FitsExtensionLabelSchema
    frame_path: SchemaPath
    header: list[FitsHeaderSchema]
    data: FitsDataSchema


class FitsSchemaConfiguration(ABC):
    def build_fits_schema(self, frame_type: type[Frame]) -> list[FitsExtensionSchema]:
        raise NotImplementedError()

    def _walk_frame(
        self, frame_type: type[Frame], parent_header: Sequence[FitsHeaderSchema], path: SchemaPath
    ) -> list[FitsExtensionSchema]:
        frame_header = list(parent_header)
        for field_name, field_info in frame_type.frame_fields.items():
            if (
                header_schema := self._extract_frame_header_schema(field_info, SchemaPath(field_name))
            ) is not None:
                frame_header.append(header_schema)
        results: list[FitsExtensionSchema] = []
        for field_name, field_info in frame_type.frame_fields.items():
            results.extend(
                self._extract_extension_schema(field_info, frame_header, path, SchemaPath(field_name))
            )
        return results

    def _extract_frame_header_schema(
        self,
        field_info: FieldInfo,
        path_from_frame: SchemaPath,
    ) -> FitsHeaderSchema | None:
        match field_info:
            case ValueFieldInfo():
                return self.get_value_header_schema(field_info, path_from_frame)
            case HeaderFieldInfo():
                return FitsOpaqueHeaderSchema(field_info, path_from_frame)
            case MappingFieldInfo():
                return self._extract_frame_header_schema(
                    field_info.value, path_from_frame.push(Placeholders.MAPPING)
                )
        return None

    def _extract_extension_schema(
        self,
        field_info: FieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
    ) -> list[FitsExtensionSchema]:
        results: list[FitsExtensionSchema] = []
        match field_info:
            case ImageFieldInfo():
                if (
                    extension_schema := self.get_image_extension_schema(
                        field_info, frame_header, frame_path, path_from_frame
                    )
                ) is not None:
                    results.append(extension_schema)
            case MaskFieldInfo():
                if (
                    extension_schema := self.get_mask_extension_schema(
                        field_info, frame_header, frame_path, path_from_frame
                    )
                ) is not None:
                    results.append(extension_schema)
            case MappingFieldInfo():
                results.extend(
                    self._extract_extension_schema(
                        field_info.value, frame_header, frame_path, path_from_frame.push(Placeholders.MAPPING)
                    )
                )
            case FrameFieldInfo():
                results.extend(
                    self._walk_frame(field_info.cls, frame_header, frame_path.join(path_from_frame))
                )
        return results

    def get_value_header_schema(
        self, field_info: ValueFieldInfo, path_from_frame: SchemaPath
    ) -> FitsValueHeaderSchema | None:
        if field_info.fits_header is True:
            template_terms: list[str] = []
            substitutions: list[SchemaPath] = []
            for term in path_from_frame:
                match term:
                    case Placeholders.MAPPING | Placeholders.SEQUENCE:
                        template_terms.append(f"{{{len(substitutions)}}}")
                        substitutions.append(SchemaPath(*path_from_frame))
                    case _:
                        template_terms.append(term.upper())
            key = SchemaPathName(template="-".join(template_terms), substitutions=tuple(substitutions))
        elif field_info.fits_header:
            key = SchemaPathName(template=field_info.fits_header)
        else:
            return None
        return FitsValueHeaderSchema(key, field_info, path_from_frame)

    def get_extension_label_schema(self, field_info: FieldInfo, path: SchemaPath) -> FitsExtensionLabelSchema:
        raise NotImplementedError("TODO")

    def get_image_extension_schema(
        self,
        field_info: ImageFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
    ) -> FitsExtensionSchema:
        header = list(frame_header)
        header.append(FitsImageHeaderSchema(field_info, path_from_frame))
        return FitsExtensionSchema(
            self.get_extension_label_schema(field_info, frame_path.join(path_from_frame)),
            frame_path,
            header,
            FitsImageDataSchema(field_info, path_from_frame),
        )

    def get_mask_extension_schema(
        self,
        field_info: MaskFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
    ) -> FitsExtensionSchema:
        header = list(frame_header)
        header.append(FitsMaskHeaderSchema(field_info, path_from_frame))
        return FitsExtensionSchema(
            self.get_extension_label_schema(field_info, frame_path.join(path_from_frame)),
            frame_path,
            header,
            FitsMaskDataSchema(field_info, path_from_frame),
        )
