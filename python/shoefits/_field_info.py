from __future__ import annotations

__all__ = (
    "ValueFieldInfo",
    "ImageFieldInfo",
    "MaskFieldInfo",
    "StructFieldInfo",
    "MappingFieldInfo",
    "SequenceFieldInfo",
    "ModelFieldInfo",
    "HeaderFieldInfo",
    "FieldInfo",
)
import dataclasses
import types
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, Union, cast, final, get_args, get_origin

import astropy.io.fits
import numpy.typing as npt
import pydantic

from . import asdf_utils
from ._compression import FitsCompression
from ._dtypes import NumberType, UnsignedIntegerType, ValueType, numpy_to_str
from ._geom import Box

if TYPE_CHECKING:
    from ._image import Image
    from ._mask import Mask, MaskPlane
    from ._struct import Field, Struct


_R = TypeVar("_R")


def no_bbox(factory: Callable[[], _R]) -> Callable[[Box | None], _R]:
    def wrapper(bbox: Box | None) -> _R:
        return factory()

    return wrapper


@dataclasses.dataclass(kw_only=True)
class FieldInfoBase:
    description: str = ""
    allow_none: bool = False

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        if self.allow_none:
            return None
        raise TypeError(f"No default provided for {struct_type.__name__}.{name}.")


@final
@dataclasses.dataclass(kw_only=True)
class ValueFieldInfo(FieldInfoBase):
    type_name: ValueType
    unit: asdf_utils.Unit | None = None
    fits_header: bool | str = False
    default: int | str | float | None = None

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        annotation: type[object],
        **kwargs: Any,
    ) -> ValueFieldInfo:
        if annotation not in (int, str, float):
            raise TypeError(
                f"Invalid type {annotation.__name__} for value field {struct_type.__name__}.{name}."
            )
        return cls(type_name=cast(ValueType, annotation.__name__), **kwargs)

    @pydantic.model_validator(mode="after")
    def _validate_default(self) -> ValueFieldInfo:
        if self.default is not None and type(self.default).__name__ != self.type_name:
            raise TypeError(f"Default value {self.default!r} is the wrong type for {self.type_name}.")
        return self

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        if self.default is not None:
            return self.default
        return super().get_default(struct_type, name, parent_bbox)


@final
@dataclasses.dataclass(kw_only=True)
class ImageFieldInfo(FieldInfoBase):
    type_name: NumberType
    unit: asdf_utils.Unit | None = None
    use_parent_bbox: bool = True
    fits_image_extension: bool | str = True
    default: int | float | None

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        annotation: type[Image],
        *,
        dtype: npt.DTypeLike,
        allow_none: bool = False,
        **kwargs: Any,
    ) -> ImageFieldInfo:
        type_name = numpy_to_str(dtype, NumberType)
        # Use zero as the default fill value if setting the whole field to None
        # is not allowed.
        if "default" not in kwargs and not allow_none:
            kwargs["default"] = 0
        return cls(type_name=type_name, allow_none=allow_none, **kwargs)

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        from ._image import Image

        if self.default is not None and self.use_parent_bbox:
            return Image(self.default, bbox=parent_bbox, unit=self.unit, dtype=self.type_name)
        return super().get_default(struct_type, name, parent_bbox)


@final
@dataclasses.dataclass(kw_only=True)
class MaskFieldInfo(FieldInfoBase):
    type_name: UnsignedIntegerType = "uint8"
    required_planes: list[MaskPlane] = pydantic.Field(default_factory=list)
    allow_additional_planes: bool = True
    use_parent_bbox: bool = True
    fits_image_extension: bool | str = True
    fits_plane_header_style: Literal["afw"] | None = "afw"
    fits_compression: FitsCompression | None = None
    default: int | None

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        annotation: type[Mask],
        *,
        dtype: npt.DTypeLike,
        allow_none: bool = False,
        **kwargs: Any,
    ) -> MaskFieldInfo:
        type_name = numpy_to_str(dtype, NumberType)
        # Use zero as the default fill value if setting the whole field to None
        # is not allowed.
        if "default" not in kwargs and not allow_none:
            kwargs["default"] = 0
        return cls(type_name=type_name, allow_none=allow_none, **kwargs)

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        from ._mask import Mask, MaskSchema

        if self.default is not None and self.use_parent_bbox:
            schema = MaskSchema(self.required_planes, dtype=self.type_name)
            return Mask(self.default, bbox=parent_bbox, schema=schema)
        return super().get_default(struct_type, name, parent_bbox)


@final
@dataclasses.dataclass(kw_only=True)
class StructFieldInfo(FieldInfoBase):
    cls: type[Struct]
    is_frame: bool
    use_parent_bbox: bool = True
    default_factory: Callable[[Box | None], Struct] | None

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        annotation: type[Struct],
        *,
        allow_none: bool = False,
        **kwargs: Any,
    ) -> StructFieldInfo:
        from ._frame import Frame

        if "default_factory" not in kwargs and not allow_none:
            # If we weren't given a default_factory and can't initialize to
            # None, try to use the class constructor as the default factory.
            kwargs["default_factory"] = kwargs["cls"]
        return cls(cls=annotation, is_frame=issubclass(annotation, Frame), **kwargs)

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        if self.default_factory is not None:
            if self.is_frame and not self.use_parent_bbox:
                parent_bbox = None
            return self.default_factory(parent_bbox)
        return super().get_default(struct_type, name, parent_bbox)


@final
@dataclasses.dataclass(kw_only=True)
class MappingFieldInfo(FieldInfoBase):
    cls: type[Mapping]
    value: FieldInfo
    default_factory: Callable[[Box | None], Mapping[str, Any]] | None

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        origin_type: type[Mapping[str, Any]],
        annotation: Any,
        *,
        default_factory: Callable[[Box | None], Mapping[str, Any]] | None = None,
        value: Field | None = None,
        **kwargs: Any,
    ) -> MappingFieldInfo:
        key_type, value_type = get_args(annotation)
        if key_type is not str:
            raise TypeError(
                f"Key type for mapping field {struct_type.__name__}.{name} must be 'str', not {key_type}."
            )
        if value is not None:
            kwargs.update(value._kwargs)
        return cls(
            cls=origin_type,
            value=_build_field_info(struct_type, name, value_type, kwargs),
            default_factory=default_factory,
        )

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        if self.default_factory is not None:
            return self.default_factory(parent_bbox)
        return super().get_default(struct_type, name, parent_bbox)


@final
@dataclasses.dataclass(kw_only=True)
class SequenceFieldInfo(FieldInfoBase):
    cls: type[Sequence]
    value: FieldInfo
    default_factory: Callable[[Box | None], Sequence] | None

    @classmethod
    def build(
        cls,
        name: str,
        struct_type: type[Struct],
        origin_type: type[Sequence[Any]],
        annotation: Any,
        *,
        default_factory: Callable[[Box | None], Sequence] | None = None,
        value: Field | None = None,
        **kwargs: Any,
    ) -> SequenceFieldInfo:
        value_type = get_args(annotation)
        if value is not None:
            kwargs.update(value._kwargs)
        return cls(
            cls=origin_type,
            value=_build_field_info(struct_type, name, value_type, kwargs),
            default_factory=default_factory,
        )

    def get_default(self, struct_type: type[Struct], name: str, parent_bbox: Box | None) -> Any:
        if self.default_factory is not None:
            return self.default_factory(parent_bbox)
        return super().get_default(struct_type, name, parent_bbox)


@final
@dataclasses.dataclass(kw_only=True)
class ModelFieldInfo(FieldInfoBase):
    cls: type[pydantic.BaseModel]
    default_factory: Callable[[Box | None], pydantic.BaseModel] | None

    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[pydantic.BaseModel], **kwargs: Any
    ) -> ModelFieldInfo:
        return cls(cls=annotation, **kwargs)


@final
@dataclasses.dataclass(kw_only=True)
class HeaderFieldInfo(FieldInfoBase):
    @classmethod
    def build(
        cls, name: str, struct_type: type[Struct], annotation: type[astropy.io.fits.Header], **kwargs: Any
    ) -> HeaderFieldInfo:
        return cls(**kwargs)


FieldInfo: TypeAlias = Union[
    ValueFieldInfo,
    ImageFieldInfo,
    MaskFieldInfo,
    MappingFieldInfo,
    SequenceFieldInfo,
    ModelFieldInfo,
    HeaderFieldInfo,
    StructFieldInfo,
]


def _build_field_info(
    struct_type: type[Struct],
    name: str,
    annotation: Any,
    kwargs: dict[str, Any],
    annotation_had_none_stripped: bool = False,
) -> FieldInfo:
    from ._image import Image
    from ._mask import Mask
    from ._struct import Struct

    if isinstance(annotation, type):
        if kwargs.get("allow_none", False) and not annotation_had_none_stripped:
            raise TypeError(
                "Annotation does not permit None, but allow_none=True was set explicitly "
                f"on field {struct_type.__name__}.{name}."
            )
        if annotation is Image:
            return ImageFieldInfo.build(name, struct_type, annotation, **kwargs)
        if annotation is Mask:
            return MaskFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, Struct):
            return StructFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, pydantic.BaseModel):
            return ModelFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, astropy.io.fits.Header):
            return HeaderFieldInfo.build(name, struct_type, annotation, **kwargs)
        if issubclass(annotation, Mapping):
            raise TypeError(f"Mapping field {struct_type.__name__}.{name!r} must have a type annotations.")
        return ValueFieldInfo.build(name, struct_type, annotation, **kwargs)
    origin_type = get_origin(annotation)
    if issubclass(origin_type, Mapping):
        return MappingFieldInfo.build(
            name, struct_type, cast(type[Mapping[str, Any]], origin_type), annotation, **kwargs
        )
    if issubclass(origin_type, Sequence):
        return SequenceFieldInfo.build(
            name, struct_type, cast(type[Sequence[Any]], origin_type), annotation, **kwargs
        )
    if origin_type is types.UnionType:
        match get_args(annotation):
            case (optional_type, types.NoneType) | (types.NoneType, optional_type):
                if not kwargs.setdefault("allow_none", True):
                    raise TypeError(
                        "Annotation permits None, but allow_none=False was set explicitly "
                        f"on field {struct_type.__name__}.{name}."
                    )
                return _build_field_info(
                    struct_type, name, optional_type, kwargs, annotation_had_none_stripped=True
                )
    raise TypeError(f"Unsupported type {annotation} for field {struct_type.__name__}.{name}.")
