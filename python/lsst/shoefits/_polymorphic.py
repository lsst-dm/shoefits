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

__all__ = (
    "PolymorphicAdapter",
    "PolymorphicAdapterRegistry",
    "GetPolymorphicTag",
    "PolymorphicReadError",
    "PolymorphicWriteError",
    "Polymorphic",
)

import warnings
from abc import abstractmethod
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import astropy.io.fits
import pydantic
import pydantic_core.core_schema as pcs

from ._read_context import ReadContext, ReadError
from ._write_context import WriteContext, WriteError
from .json_utils import JsonValue

_C = TypeVar("_C", bound=type[Any])
_T = TypeVar("_T")
_S = TypeVar("_S", bound=pydantic.BaseModel)


class GetPolymorphicTag(Protocol):
    """Interface for callables used to get the tag string for a polymorphic
    object.
    """

    def __call__(self, instance: Any) -> str: ...


class PolymorphicAdapter(Generic[_T, _S]):
    """A helper class that transforms arbitrary objects to Pydantic models.

    Adapter instances are associated with string tags in a
    `PolymorphicAdapterRegistry`, and are used serializing or serializing an
    object in a `Polymorphic` annotated field.

    Notes
    -----
    `Polymorphic` has two generic type parameters:

    - the arbitrary type being adapted
    - the Pydantic model type it is mapped to for serialization.
    """

    @property
    @abstractmethod
    def model_type(self) -> type[_S]:
        """The Pydantic model type the polymorphic type is mapped to for
        serialization.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_model(self, polymorphic: _T, /) -> _S:
        """Convert a polymorphic object to a Pydantic model instance."""
        raise NotImplementedError()

    @abstractmethod
    def from_model(self, model: _S, /) -> _T:
        """Construct a polymorphic object from a Pydantic model instance."""
        raise NotImplementedError()

    @property
    def nest(self) -> bool:
        """Whether to consider this type a logical level of nesting in the
        serialized form.

        For FITS serialization, this controls whether FITS headers exported by
        this type or fields within it are included only in HDUs generated from
        this type's fields (``nest=True``, default) vs. included also in parent
        or sibling HDUs (``nest=False``).  It also increments the EXTLEVEL
        header value for nested HDUs.
        """
        return True

    def extract_fits_header(self, polymorphic: _T) -> astropy.io.fits.Header | None:
        """Return a FITS header values that should be exported to any HDUs
        associated with the polymorphic object when it is saved.

        Notes
        -----
        The default implementation returns `None`, resulting in no FITS header
        values being exported.  FITS header exports are stripped on read, not
        read (see `strip_fits_header`).

        See `FitsWriteContext` for details of how FITS HDUs are associated with
        objects in a nested data structure.
        """
        return None

    def strip_fits_header(self, header: astropy.io.fits.Header) -> None:
        """Strip any FITS header keys that may have been added by this adapter
        on write.

        Notes
        -----
        This method is not guaranteed to be called, as in some cases a FITS
        header may be completely discarded instead (situations where we
        actually read from FITS headers are very rare; we prefer to duplicate
        information in the object tree itself instead).
        """
        pass


class NativeAdapter(PolymorphicAdapter[_S, _S]):
    """An implementation of `PolymorphicAdapter` for polymorphic objects that
    are themselves Pydantic models.
    """

    def __init__(self, native_type: type[_S]):
        self._native_type = native_type

    @property
    def model_type(self) -> type[_S]:
        return self._native_type

    def to_model(self, polymorphic: _S) -> _S:
        return polymorphic

    def from_model(self, model: _S) -> _S:
        return model


class PolymorphicAdapterRegistry:
    """A registry of `PolymorphicAdapter` objects that is used to save and load
    objects in `Polymorphic` annotated fields.

    Notes
    -----
    This registry provides a way for the [de]serialization machinery to go from
    a string tag to `PolymorphicAdapter` instance.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, PolymorphicAdapter[Any, pydantic.BaseModel]] = {}

    def __getitem__(self, tag: str) -> PolymorphicAdapter[Any, pydantic.BaseModel]:
        return self._adapters[tag]

    def register_adapter(self, tag: str, adapter: PolymorphicAdapter[Any, Any]) -> None:
        """Register the given adapter instance with the given string tag."""
        self._adapters[tag] = adapter

    def register_native(self, tag: str, native_type: type[Any]) -> None:
        """Register the given type with the given string tag as a native type
        that is already a Pydantic model.
        """
        self._adapters[tag] = NativeAdapter(native_type)


class PolymorphicReadError(ReadError):
    """Exception raised by the serialization machinery when it cannot load
    the object for a `Polymorphic` annotated field.
    """


class PolymorphicWriteError(WriteError):
    """Exception raised by the serialization machinery when it cannot save
    the object in a `Polymorphic` annotated field.
    """


class Polymorphic:
    """An annotation for Pydantic fields for load-time polymorphism.

    Parameters
    ----------
    get_tag
        Callable used to obtain the string tag that identifies a concrete type
        from an instance.
    on_load_failure, optional
        How to handle read attempts in which the string tag that was serialized
        is not present in the `PolymorphicAdapterRegistry`.  Note that other
        read failures (which typically represent bugs or malformed
        serialization outputs) are always exceptions.

    Notes
    -----
    This class is designed to be used with `typing.Annotated` as part of the
    type annotation for a Pydantic model field, and it must be accompanied by
    additional declarations on the concrete types that can be serialized::

        from abc import ABC from typing import Annotated import shoefits as shf


        class Base(ABC):
            '''Base class for polymorphic hierarchy (not concrete).'''


        class Derived1(Base):
            '''A concrete implementation the serialization author has no
            control over. '''

            def __init__(self, x: str) -> None:
                self.x = x


        class Model1(pydantic.BaseModel):
            ```Model used for serialization of `Derived1` instances.```

            x: str


        class Adapter1(shf.PolymorphicAdapter[Derived1, Model1]):
            ```Adapter that maps `Derived1` to `Model1` for serialization.```

            @property def model_type(self) -> type[Model1]:
                return Model1

            def to_model(self, d: Derived1) -> Model1:
                return Model1(x=d.x)

            def from_model(self, m: Model1) -> Derived1:
                return Derived1(x=m.x)


        class Derived2(Base, pydantic.BaseModel):
            '''A concrete implementation that is itself a Pydantic model.'''

            y: int

            tag: ClassVar[str] = "derived2"


        class Holder(pydantic.BaseModel):
            '''A model with a polymorphic field.'''

            p: Annotated[
                Base,
                Polymorphic(lambda x: getattr(x, "tag", "derived1")),
            ]


    Like any other use of `typing.Annotated`, to static type checkers, the type
    of ``Holder.p``  is simply `Base`.  For serialization, only objects whose
    types have been registered with tags may be saved and loaded (unless tag
    extraction is customized via the ``get_tag`` argument, and there is no
    runtime check that loaded objects actually inherit from the annotated type
    (which allows it to be a `Protocol` or `typing.Union` rather than a true
    base class).

    In addition, adapters must be associated with each tag via a
    `PolymorphicAdapterRegistry`.  How a registry is populated in practice and
    passed to serialization and serialization code (subclasses of
    `WriteContext` and `ReadContext`) is left to downstream applications.  For
    this example, this amounts to something like::

        adapter_registry = shf.PolymorphicAdapterRegistry()
        adapter_registry.register_adapted("derived1", Adapter1())
        adapter_registry.register_native("derived2", Derived2)


    When an adapter cannot be found for a serialized tag, the default behavior
    is to raise `PolymorphicReadError`.  Polymorphic fields can also be told to
    ignore these failures or warn about them, but only if they are annotated as
    possibly being `None`::

        p: Annotated[Base | None, Polymorphic(..., on_load_failure="warn")]

    If a field may be `None` but load failures should be treated as errors, the
    nullable annotation should appear outside `typing.Annotated`::

        p: Annotated[Base, Polymorphic(..., on_load_failure="raise")] | None

    """

    def __init__(
        self,
        get_tag: GetPolymorphicTag,
        on_load_failure: Literal["raise", "warn", "ignore"] = "raise",
    ):
        self.get_tag = get_tag
        self.on_load_failure = on_load_failure

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        nullable = False
        if get_origin(source_type) is Union and None in get_args(source_type):
            nullable = True
        if self.on_load_failure == "raise":
            if nullable:
                raise TypeError(
                    "Polymorphic annotations with on_load_failure='raise' should not include None; "
                    "use 'Annotated[T, Polymorphic(...)] | None' instead of "
                    "'Annotated[T | None, Polymorphic(...)]'"
                )
        elif not nullable:
            raise TypeError(
                f"Polymorphic annotations with on_load_failure='{self.on_load_failure}' should include None; "
                "use Annotated[T | None, Polymorphic(...)]' instead of "
                "'Annotated[T, Polymorphic(...)] | None'"
            )
        from_json_schema: pcs.CoreSchema = pcs.chain_schema(
            [
                pcs.dict_schema(pcs.str_schema(), pcs.any_schema()),
                pcs.with_info_plain_validator_function(self.deserialize),
            ]
        )
        from_python_schema: pcs.CoreSchema = pcs.tagged_union_schema(
            {"tagged_dict": from_json_schema, "instance": pcs.any_schema()},
            self._deserialize_discriminator,
        )
        if nullable:
            from_json_schema = pcs.nullable_schema(from_json_schema)
            from_python_schema = pcs.nullable_schema(from_python_schema)
            serializer = self.serialize_maybe_null
        else:
            serializer = self.serialize_not_null
        return pcs.json_or_python_schema(
            json_schema=from_json_schema,
            python_schema=from_python_schema,
            serialization=pcs.plain_serializer_function_ser_schema(serializer, info_arg=True),
        )

    def deserialize(self, tree: dict[str, JsonValue] | None, info: pydantic.ValidationInfo) -> Any:
        if tree is None:
            return None
        if (read_context := ReadContext.from_info(info)) is None:
            raise PolymorphicReadError(
                "Polymorphic fields require a ReadContext in the Pydantic validation context."
            )
        match tree:
            case {"$tag": str(tag)}:
                pass
            case _:
                raise PolymorphicReadError("No string '$tag' entry found for polymorphic field.")
        try:
            adapter: PolymorphicAdapter[Any, Any] = read_context.polymorphic_adapter_registry[tag]
        except KeyError as err:
            if self.on_load_failure == "raise":
                raise PolymorphicReadError(str(err))
            if self.on_load_failure == "warn":
                warnings.warn(str(err))
            return None
        model_type: type[pydantic.BaseModel] = adapter.model_type
        if "$tag" not in model_type.model_fields:
            del tree["$tag"]
        if header := read_context.primary_header:
            adapter.strip_fits_header(header)
        serialized = model_type.model_validate(tree, context=info.context)
        return adapter.from_model(serialized)

    def serialize_maybe_null(self, obj: Any, info: pydantic.SerializationInfo) -> dict[str, JsonValue] | None:
        if obj is None:
            return None
        return self.serialize_not_null(obj, info)

    def serialize_not_null(self, obj: Any, info: pydantic.SerializationInfo) -> dict[str, JsonValue]:
        if (write_context := WriteContext.from_info(info)) is None:
            raise PolymorphicWriteError(
                "Polymorphic field writes require an adapter registry in the Pydantic serialization context."
            )
        tag = self.get_tag(obj)
        adapter = write_context.polymorphic_adapter_registry[tag]
        serialized = adapter.to_model(obj)
        data = serialized.model_dump(context=info.context)
        if data.setdefault("$tag", tag) != tag:
            raise PolymorphicWriteError(
                f"Serialized form already has $tag={data['tag']!r}, "
                f"which is inconsistent with $tag={tag!r} from the get_tag callback."
            )
        if extracted := adapter.extract_fits_header(obj):
            write_context.export_fits_header(extracted)
        return data

    @staticmethod
    def _deserialize_discriminator(obj: Any) -> str:
        match obj:
            case {"$tag": _}:
                return "tagged_dict"
            case _:
                return "instance"
