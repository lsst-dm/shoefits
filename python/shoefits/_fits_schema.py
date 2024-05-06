from __future__ import annotations

__all__ = (
    "FitsHeaderSchema",
    "FitsValueHeaderSchema",
    "FitsImageHeaderSchema",
    "FitsColumnSchema",
    "FitsDataSchema",
    "FitsBinaryTableSchema",
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


@dataclasses.dataclass
class FitsColumnSchema:
    name: SchemaPathName
    field_info: ValueFieldInfo | ImageFieldInfo | MaskFieldInfo
    path_from_row: SchemaPath


class FitsDataSchema(ABC):
    pass


@dataclasses.dataclass
class FitsBinaryTableSchema(FitsDataSchema):
    table_path_from_frame: SchemaPath
    """Path of the container whose whose items make up the rows of this table,
    relative to `FitsExtensionSchema.frame_path`.
    """

    key_column_name: str | None

    columns: list[FitsColumnSchema] = dataclasses.field(default_factory=list)

    @property
    def row_path_from_frame(self) -> SchemaPath:
        """Path that corresponds to a row of this table, which always maps to
        a `Frame`.
        """
        return self.table_path_from_frame.push(Placeholders.MAPPING)


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
    extlevel: int
    extver: SchemaPath | None = None


@dataclasses.dataclass
class FitsExtensionSchema:
    label: FitsExtensionLabelSchema
    frame_path: SchemaPath
    header: list[FitsHeaderSchema]
    data: FitsDataSchema


class FitsSchemaConfiguration(ABC):
    def build_fits_schema(self, frame_type: type[Frame]) -> list[FitsExtensionSchema]:
        return self._walk_frame(frame_type, [], SchemaPath(), 1)

    def _walk_frame(
        self,
        frame_type: type[Frame],
        parent_header: Sequence[FitsHeaderSchema],
        path: SchemaPath,
        extlevel: int,
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
                self._extract_extension_schema(
                    field_info, frame_header, path, SchemaPath(field_name), extlevel
                )
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
        extlevel: int,
    ) -> list[FitsExtensionSchema]:
        results: list[FitsExtensionSchema] = []
        match field_info:
            case ImageFieldInfo():
                if (
                    extension_schema := self.get_image_extension_schema(
                        field_info,
                        frame_header,
                        frame_path=frame_path,
                        path_from_frame=path_from_frame,
                        extlevel=extlevel,
                    )
                ) is not None:
                    results.append(extension_schema)
            case MaskFieldInfo():
                if (
                    extension_schema := self.get_mask_extension_schema(
                        field_info,
                        frame_header,
                        frame_path=frame_path,
                        path_from_frame=path_from_frame,
                        extlevel=extlevel,
                    )
                ) is not None:
                    results.append(extension_schema)
            case MappingFieldInfo():
                results.extend(
                    self.get_mapping_extensions_schema(
                        field_info,
                        frame_header,
                        frame_path=frame_path,
                        path_from_frame=path_from_frame,
                        extlevel=extlevel,
                    )
                )
            case FrameFieldInfo():
                results.extend(
                    self._walk_frame(
                        field_info.cls, frame_header, frame_path.join(path_from_frame), extlevel=extlevel + 1
                    )
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

    def get_extension_label_schema(
        self, path_from_frame: SchemaPath, extlevel: int
    ) -> FitsExtensionLabelSchema:
        raise NotImplementedError("TODO")

    def get_image_extension_schema(
        self,
        field_info: ImageFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        extlevel: int,
    ) -> FitsExtensionSchema | None:
        if not field_info.fits_image_extension:
            return None
        elif field_info.fits_image_extension is not True:  # i.e. it's a string
            label = FitsExtensionLabelSchema(
                extname=SchemaPathName(field_info.fits_image_extension), extlevel=extlevel
            )
        else:
            label = self.get_extension_label_schema(frame_path.join(path_from_frame), extlevel)
        header = list(frame_header)
        header.append(FitsImageHeaderSchema(field_info, path_from_frame))
        return FitsExtensionSchema(
            label,
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
        extlevel: int,
    ) -> FitsExtensionSchema | None:
        if not field_info.fits_image_extension:
            return None
        elif field_info.fits_image_extension is not True:  # i.e. it's a string
            label = FitsExtensionLabelSchema(
                extname=SchemaPathName(field_info.fits_image_extension), extlevel=extlevel
            )
        else:
            label = self.get_extension_label_schema(frame_path.join(path_from_frame), extlevel)
        header = list(frame_header)
        header.append(FitsMaskHeaderSchema(field_info, path_from_frame))
        return FitsExtensionSchema(
            label,
            frame_path,
            header,
            FitsMaskDataSchema(field_info, path_from_frame),
        )

    def get_mapping_extensions_schema(
        self,
        field_info: MappingFieldInfo,
        frame_header: Sequence[FitsHeaderSchema],
        frame_path: SchemaPath,
        path_from_frame: SchemaPath,
        extlevel: int,
    ) -> list[FitsExtensionSchema]:
        if field_info.fits_table_extension:
            if field_info.fits_table_extension is not True:  # i.e. it's a string
                label = FitsExtensionLabelSchema(
                    extname=SchemaPathName(field_info.fits_table_extension), extlevel=extlevel
                )
            else:
                label = self.get_extension_label_schema(frame_path.join(path_from_frame), extlevel)
            assert isinstance(field_info.value, FrameFieldInfo), "Guaranteed by MappingFieldInfo validator."
            columns: list[FitsColumnSchema] = []
            for column_field_name, column_field_info in field_info.value.cls.frame_fields.items():
                columns.extend(self._extract_column_schema(column_field_info, SchemaPath(column_field_name)))
            return [
                FitsExtensionSchema(
                    label=label,
                    frame_path=frame_path,
                    header=list(frame_header),
                    data=FitsBinaryTableSchema(path_from_frame, field_info.fits_table_key_column, columns),
                )
            ]
        return self._extract_extension_schema(
            field_info.value,
            frame_header,
            frame_path,
            path_from_frame.push(Placeholders.MAPPING),
            extlevel,
        )

    def _extract_column_schema(
        self, field_info: FieldInfo, path_from_row: SchemaPath
    ) -> list[FitsColumnSchema]:
        results: list[FitsColumnSchema] = []
        match field_info:
            case ValueFieldInfo() | ImageFieldInfo() | MaskFieldInfo():
                if (column_schema := self.get_column_schema(field_info, path_from_row)) is not None:
                    results.append(column_schema)
            case FrameFieldInfo():
                for nested_name, nested_field_info in field_info.cls.frame_fields.items():
                    results.extend(
                        self._extract_column_schema(nested_field_info, path_from_row.push(nested_name))
                    )
            case MappingFieldInfo():
                results.extend(
                    self._extract_column_schema(field_info.value, path_from_row.push(Placeholders.MAPPING))
                )
        return results

    def get_column_name_schema(self, path_from_row: SchemaPath) -> SchemaPathName:
        raise NotImplementedError("TODO")

    def get_column_schema(
        self, field_info: ValueFieldInfo | ImageFieldInfo | MaskFieldInfo, path_from_row: SchemaPath
    ) -> FitsColumnSchema | None:
        if not field_info.fits_column:
            return None
        if field_info.fits_column is True:
            name = self.get_column_name_schema(path_from_row)
        else:
            name = SchemaPathName(field_info.fits_column)
        return FitsColumnSchema(name, field_info, path_from_row)


# TODO:
# - mapping keys, image shapes from configuration; identify which ones need
#   consistency (e.g. mappings inside binary table rows).
# - override methods for all FieldInfo fits_* options.
# - grids (including emitting binary tables for extras)
# - options for dropping binary table exports from tree
