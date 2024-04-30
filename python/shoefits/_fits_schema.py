from __future__ import annotations

__all__ = ("FitsHeaderKeySchema", "FitsExtensionLabelSchema", "FitsExtensionSchema")


import dataclasses

from ._field import FieldInfo
from ._schema_path import Placeholders, SchemaPath, SchemaPathTerm


@dataclasses.dataclass
class FitsHeaderKeySchema:
    path: SchemaPath
    info: FieldInfo
    key_pattern: str
    key_substitutions: list[SchemaPath]

    @classmethod
    def from_path(cls, path: SchemaPath, info: FieldInfo) -> FitsHeaderKeySchema:
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
        return cls(
            path=path,
            info=info,
            key_pattern="-".join(key_pattern_terms),
            key_substitutions=key_substitutions,
        )


@dataclasses.dataclass
class FitsExtensionLabelSchema:
    extname_pattern: str
    extname_substitutions: list[SchemaPath]
    extver_substitution: SchemaPath | None

    @classmethod
    def from_path(cls, path: SchemaPath) -> FitsExtensionLabelSchema:
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
        return cls(
            extname_pattern="/".join(extname_pattern_terms),
            extname_substitutions=extname_substitutions,
            extver_substitution=extver_substitution,
        )


@dataclasses.dataclass
class FitsExtensionSchema:
    label: FitsExtensionLabelSchema
    frame_path: SchemaPath
    data_path: SchemaPath
    data_info: FieldInfo
    header: list[FitsHeaderKeySchema]
