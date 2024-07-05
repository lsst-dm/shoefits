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
    "register_tag",
    "get_tag_from_registry",
)

import os
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
    overload,
)

import astropy.io.fits
import pydantic
import pydantic_core.core_schema as pcs

from lsst.resources import ResourcePath
from lsst.utils.doImport import doImportType

from ._read_context import ReadContext, ReadError
from ._write_context import WriteContext, WriteError
from .json_utils import JsonValue

_C = TypeVar("_C", bound=type[Any])
_T = TypeVar("_T")
_S = TypeVar("_S", bound=pydantic.BaseModel)


CONFIGS_ENVVAR: str = "SHOEFITS_ADAPTER_CONFIGS"

_tag_registry: dict[type[Any], str] = {}


class GetPolymorphicTag(Protocol):
    """Interface for callables used to get the tag string for a polymorphic
    object.
    """

    def __call__(self, instance: Any) -> str: ...


def get_tag_from_registry(instance: Any) -> str:
    """Get the tag string for a polymorphic object from the singleton tag
    registry.

    Notes
    -----
    This function uses strict type equality, not inheritance relationships, so
    it can only be used in cases where the exact type of the given object has
    been registered via `register_tag`, not just a base class.
    """
    return _tag_registry[type(instance)]


@overload
def register_tag(tag: str) -> Callable[[_C], _C]: ...


@overload
def register_tag(tag: str, target_type: _C) -> _C: ...


def register_tag(tag: str, target_type: _C | None = None) -> Any:
    """Register a type for participation in a `Polymorphic` annotated field.

    Notes
    -----
    This function can be used as a decorator::

        @register_tag("example_a")
        class ExampleA:
            ...

    or called with the `type` to register as its second argument::

        class ExampleA:
            ...

        register_tag("example_a", ExampleA)

    Registering the exact type (not just a base class) allows the default
    `get_tag_from_registry` function to be used to obtain the string tag for
    an object when serializing a `Polymorphic` annotated field.  This is only
    one of the two registrations needed by the `Polymorphic` system; see
    `PolymorphicAdapterRegistry` for the other.
    """
    if target_type is None:

        def decorator(target_type: _C) -> _C:
            _tag_registry[target_type] = tag
            return target_type

        return decorator

    else:
        _tag_registry[target_type] = tag
        return target_type


@final
class AdapterFactory(pydantic.BaseModel):
    """A serializable factory for `PolymorphicAdapter` objects.

    The adapter factory model represents the values of an adapter registry
    file, which is a dictionary-like JSON file with tag strings as keys.  See
    `PolymorphicAdapterRegistry` for details.  For adapters without any
    constructor arguments or native polymorphic types (which are already
    Pydantic models), simple string values are used.
    """

    type_: str = pydantic.Field(alias="type")
    """Fully-qualified class name of the adapter type.
    """

    args: list[int | str | float | bool | None] = pydantic.Field(default_factory=list)
    """Positional arguments to the adapter constructor."""

    kwargs: dict[str, int | str | float | bool | None] = pydantic.Field(default_factory=dict)
    """Keyword arguments to the adapter constructor."""

    def make_adapter(self) -> PolymorphicAdapter[Any, Any]:
        """Construct an adapter instance from this configuration."""
        type_ = doImportType(self.type_)
        if issubclass(type_, PolymorphicAdapter):
            return type_(*self.args, **self.kwargs)
        elif issubclass(type_, pydantic.BaseModel):
            return NativeAdapter(type_)
        else:
            raise TypeError(
                f"Cannot construct a PolymorphicAdapter from type {self.type_} that is neither an adapter "
                "subclass or a Pydantic model."
            )


AdapterConfig: TypeAlias = str | AdapterFactory


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
    While the `register_adapter` and `register_native` methods may be used to
    populate the registry directly, when loading an object may require
    importing a module, it is recommended that adapter definitions are instead
    declared in configuration files and the paths of those files added to the
    ``SHOEFITS_ADAPTER_CONFIGS`` environment variable.  Config files are JSON
    dictionaries, with string tags as keys and nested dictionaries
    corresponding to the `AdapterFactory` model as values.

    `PolymorphicAdapterRegistry` is not a singleton, though applications may
    wish to construct a single registry at import time.  The set of all config
    files is read from the ``SHOEFITS_ADAPTER_CONFIGS`` environment variable
    when a registry is constructed (later updates to the environment variable
    are ignored), and individual config files are read as needed (in the order
    specified in the environment variable) and their contents cached.

    This registry provides a way for the [de]serialization machinery to go from
    a string tag to `PolymorphicAdapter` instance.  When saving objects, it
    also needs a way to obtain that string tag from the object being saved,
    which (but need not) involve another registry; see `register_tag` and
    `Polymorphic` for details.
    """

    def __init__(self) -> None:
        self._validator = pydantic.TypeAdapter(dict[str, AdapterConfig])
        config_path = os.environ.get("SHOEFITS_ADAPTER_CONFIGS", "")
        if config_path:
            self._config_files_unread = [ResourcePath(path) for path in config_path.split(":")]
        else:
            self._config_files_unread = []
        self._config_files_read: list[str] = []
        # PATH-like envvars are expected to be processed such that items in
        # the beginning take priority over those in the back.  We reverse the
        # list so we can use list.pop() to go from high to low priority.
        self._config_files_unread.reverse()
        self._adapter_factories: dict[str, AdapterFactory] = {}
        self._adapters: dict[str, PolymorphicAdapter[Any, pydantic.BaseModel]] = {}

    def __getitem__(self, tag: str) -> PolymorphicAdapter[Any, pydantic.BaseModel]:
        adapter: PolymorphicAdapter[Any, pydantic.BaseModel] | None = None
        if adapter := self._adapters.get(tag):
            return adapter
        if adapter_factory := self._adapter_factories.get(tag):
            adapter = adapter_factory.make_adapter()
            self._adapters[tag] = adapter
            return adapter
        while self._config_files_unread:
            config_path = self._config_files_unread.pop()
            if not config_path.exists():
                continue
            self._config_files_read.append(str(config_path))
            for key, factory in self._validator.validate_json(config_path.read()).items():
                if isinstance(factory, str):
                    factory = AdapterFactory.model_construct(adapter_type=factory)
                self._adapter_factories.setdefault(key, factory)
                if key == tag:
                    adapter = factory.make_adapter()
            if adapter is not None:
                return adapter
        raise KeyError(
            f"No adapter found for polymorphic field with tag {tag} in any of {self._config_files_read}."
        )

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
    get_tag, optional
        Callable used to obtain the string tag that identifies a concrete type
        from an instance.  The default is `get_tag_from_registry`, which
        requires all participating types to be registered with `register_tag`.
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

        shf.register_tag("derived1", Derived1)


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


        @shf.register_tag("derived2") class Derived2(Base, pydantic.BaseModel):
            '''A concrete implementation that is itself a Pydantic model.'''

            y: int


        class Holder(pydantic.BaseModel):
            '''A model with a polymorphic field.'''

            p: Annotated[Base, Polymorphic()]


    Like any other use of `typing.Annotated`, to static type checkers, the type
    is simply `Base`.  For serialization, only objects whose types have been
    registered with tags may be saved and loaded (unless tag extraction is
    customized via the ``get_tag`` argument, and there is no runtime check that
    loaded objects actually inherit from the annotated type (which allows it to
    be a `Protocol` or `typing.Union` rather than a true base class).

    In addition, adapters must be associated with each tag via a JSON config
    field whose path is included in the ``SHOEFITS_ADAPTER_CONFIGS``
    environment variable::

        {
            # Adapted types are configured with their adapter.
            "derived1": "mymodule.Adapter1",
            # Native types are configured directly.
            "derived2": "mymodule.Derived2",
        }

    Values in the config file may be strings that are fully-qualified Python
    symbols or nested dictionaries with a "type" string field and optional
    "args" and "kwargs" fields, for adapters that take arguments at
    construction.  Arguments must be simple JSON primitives (not lists or
    dicts).

    When an adapter cannot be found for a serialized tag, the default behavior
    is to raise `PolymorphicReadError`.  Polymorphic fields can also be told to
    ignore these failures or warn about them, but only if they are annotated as
    possibly being `None`::

        p: Annotated[Base | None, Polymorphic(on_load_failure="warn")]

    If a field may be `None` but load failures should be treated as errors, the
    nullable annotation should appear outside `typing.Annotated`::

        p: Annotated[Base, Polymorphic(on_load_failure="raise")] | None

    """

    def __init__(
        self,
        get_tag: GetPolymorphicTag = get_tag_from_registry,
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
            write_context.export_header_update(extracted)
        return data

    @staticmethod
    def _deserialize_discriminator(obj: Any) -> str:
        match obj:
            case {"$tag": _}:
                return "tagged_dict"
            case _:
                return "instance"
