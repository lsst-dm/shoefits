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
    "FitsSchemaConfiguration",
)


import dataclasses
from abc import ABC
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ._field import DataExportFieldInfo, is_image_field, is_value_field
from ._images import ImageFieldInfo
from ._schema_path import Placeholders, SchemaPath, SchemaPathName, SchemaPathTerm

if TYPE_CHECKING:
    from ._field_base import ValueFieldInfo
    from ._images import ImageFieldInfo
    from ._schema import FrameSchema, Schema


class FitsHeaderSchema(ABC):
    pass


@dataclasses.dataclass
class FitsValueHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS header cards exported from a simple `str`,
    `int`, or `float` value in the YAML tree.
    """

    key: SchemaPathName
    field: ValueFieldInfo
    path: SchemaPath
    """Path of the value field, relative to `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsOpaqueHeaderSchema(FitsHeaderSchema):
    """Schema definition for a bundle of FITS opaque header cards that do not
    appear in the YAML tree at all, and are typically just propagated from
    some externally-written FITS file.
    """

    # TODO: field info

    path: SchemaPath
    """Path of the opaque-header field, relative to
    `FitsExtensionSchema.frame_path`.
    """


@dataclasses.dataclass
class FitsImageHeaderSchema(FitsHeaderSchema):
    """Schema definition for FITS headers exported by all `Image` objects."""

    field: ImageFieldInfo
    path: SchemaPath
    """Path of the image field, relative to `FitsExtensionSchema.frame_path`.
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
    field: ImageFieldInfo
    path: SchemaPath
    """Path of the image field, relative to `FitsBinaryTableSchema.row_path`.

    This path is guaranteed not to hold any mapping or sequence placeholders.
    """


@dataclasses.dataclass
class FitsValueColumnSchema(FitsColumnSchema):
    name: str
    field: ValueFieldInfo
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
    field: ImageFieldInfo
    path: SchemaPath
    """Path of the `Image` that maps to this FITS extension, relative to
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
    def make_fits_schema(self, schema: Schema) -> list[FitsExtensionSchema]:
        return self.make_frame_tree_schema(SchemaPath(), schema.tree, [])

    def make_frame_tree_schema(
        self,
        root_path: SchemaPath,
        tree: Mapping[SchemaPath, FrameSchema],
        parent_header: Sequence[FitsHeaderSchema],
    ) -> list[FitsExtensionSchema]:
        result: list[FitsExtensionSchema] = []
        for frame_suffix, frame_schema in tree.items():
            if (
                table_suffix := self.get_binary_table_path(root_path, frame_suffix, frame_schema)
            ) is not None:
                result.append(
                    self.make_binary_table_schema(root_path, table_suffix, frame_suffix, frame_schema)
                )
            else:
                result.extend(
                    self.make_frame_schema(root_path.join(frame_suffix), frame_schema, parent_header)
                )
        return result

    def make_frame_schema(
        self,
        frame_path: SchemaPath,
        frame_schema: FrameSchema,
        parent_header: Sequence[FitsHeaderSchema],
    ) -> list[FitsExtensionSchema]:
        result: list[FitsExtensionSchema] = []
        common_header = self.filter_parent_header(frame_path, parent_header)
        for suffix, header_field in frame_schema.header_exports.items():
            if is_value_field(header_field) and header_field["fits_header"]:
                common_header.append(
                    FitsValueHeaderSchema(
                        self.get_header_key(suffix, header_field),
                        header_field,
                        suffix,
                    )
                )
        for suffix, data_field in frame_schema.data_exports.items():
            data_path_abs = frame_path.join(suffix)
            extension_header = self.filter_parent_header(data_path_abs, common_header)
            if (grid_path := self.get_grid_path(frame_path, suffix, data_field)) is not None:
                result.append(self.make_grid_schema(frame_path, grid_path, suffix, data_field))
            else:
                if is_image_field(data_field):
                    data_schema = FitsImageDataSchema(data_field, suffix)
                else:
                    raise NotImplementedError(f"Unexpected type for data export field: {data_field}.")
                result.append(
                    FitsExtensionSchema(
                        self.get_extension_label(data_path_abs), frame_path, extension_header, data_schema
                    )
                )
        result.extend(self.make_frame_tree_schema(frame_path, frame_schema.tree, common_header))
        return result

    def make_binary_table_schema(
        self,
        root_path: SchemaPath,
        table_suffix: SchemaPath,
        row_frame_suffix: SchemaPath,
        row_frame_schema: FrameSchema,
    ) -> FitsExtensionSchema:
        raise NotImplementedError("TODO")

    def make_grid_schema(
        self,
        root_path: SchemaPath,
        grid_suffix: SchemaPath,
        data_suffix: SchemaPath,
        data_field: DataExportFieldInfo,
    ) -> FitsExtensionSchema:
        raise NotImplementedError("TODO")

    def filter_parent_header(
        self, path: SchemaPath, parent_header: Sequence[FitsHeaderSchema]
    ) -> list[FitsHeaderSchema]:
        return list(parent_header)

    def get_binary_table_path(
        self, root_path: SchemaPath, frame_suffix: SchemaPath, frame_schema: FrameSchema
    ) -> SchemaPath | None:
        return None

    def get_grid_path(
        self, frame_path: SchemaPath, data_suffix: SchemaPath, field: DataExportFieldInfo
    ) -> SchemaPath | None:
        return None

    def get_header_key(self, path: SchemaPath, info: ValueFieldInfo) -> SchemaPathName:
        if isinstance(info["fits_header"], str):
            return SchemaPathName(
                template=info["fits_header"],
                substitutions=(),
            )
        else:
            assert info["fits_header"], "Value should not be exported to headers."
            key_pattern_terms: list[str] = []
            key_substitutions: list[SchemaPath] = []
            cumulative_path_terms: list[SchemaPathTerm] = []
            for term in path:
                match term:
                    case Placeholders.MAPPING | Placeholders.SEQUENCE:
                        key_pattern_terms.append(f"{{{len(key_substitutions)}}}")
                        key_substitutions.append(SchemaPath(*cumulative_path_terms))
                    case concrete:
                        key_pattern_terms.append(concrete.upper())
                cumulative_path_terms.append(term)
            return SchemaPathName(
                template="-".join(key_pattern_terms),
                substitutions=tuple(key_substitutions),
            )

    def get_extension_label(self, path: SchemaPath) -> FitsExtensionLabelSchema:
        extname_pattern_terms: list[str] = []
        extname_substitutions: list[SchemaPath] = []
        extver_substitution: SchemaPath | None = None
        cumulative_path_terms: list[SchemaPathTerm] = []
        for term in path:
            extver_substitution = None
            match term:
                case Placeholders.MAPPING:
                    extname_pattern_terms.append(f"{{{len(extname_substitutions)}}}")
                    extname_substitutions.append(SchemaPath(*cumulative_path_terms))
                case Placeholders.SEQUENCE:
                    cumulative_path = SchemaPath(*cumulative_path_terms)
                    extname_pattern_terms.append(f"{{{len(extname_substitutions)}}}")
                    extname_substitutions.append(cumulative_path)
                    extver_substitution = cumulative_path
                case concrete:
                    extname_pattern_terms.append(concrete)
            cumulative_path_terms.append(term)
        if extver_substitution is not None:
            extname_pattern_terms.pop()
            extname_substitutions.pop()
        return FitsExtensionLabelSchema(
            extname=SchemaPathName(
                template="/".join(extname_pattern_terms),
                substitutions=tuple(extname_substitutions),
            ),
            extver=extver_substitution,
            extlevel=len(path),
        )
