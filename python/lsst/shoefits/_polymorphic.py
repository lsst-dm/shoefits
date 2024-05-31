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
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar, final, overload

import astropy.io.fits
import pydantic
import pydantic_core.core_schema as pcs

from lsst.resources import ResourcePath
from lsst.utils.doImport import doImportType

from .json_utils import JsonValue

if TYPE_CHECKING:
    from . import asdf_utils


_C = TypeVar("_C", bound=type[Any])
_T = TypeVar("_T")
_S = TypeVar("_S", bound=pydantic.BaseModel)


CONFIGS_ENVVAR: str = "SHOEFITS_ADAPTER_CONFIGS"

_tag_registry: dict[type[Any], str] = {}


class GetPolymorphicTag(Protocol):
    def __call__(self, instance: Any) -> str: ...


def get_tag_from_registry(instance: Any) -> str:
    return _tag_registry[type(instance)]


@overload
def register_tag(tag: str) -> Callable[[_C], _C]: ...


@overload
def register_tag(tag: str, target_type: _C) -> _C: ...


def register_tag(tag: str, target_type: _C | None = None) -> Any:
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
    target_type: str
    args: list[int | str | float | None] = pydantic.Field(default_factory=list)
    kwargs: dict[str, int | str | float | None] = pydantic.Field(default_factory=dict)

    def make_adapter(self) -> PolymorphicAdapter[Any, Any]:
        adapter_type = doImportType(self.target_type)
        return adapter_type(*self.args, **self.kwargs)


AdapterConfig: TypeAlias = str | AdapterFactory


class PolymorphicAdapter(Generic[_T, _S]):
    @property
    @abstractmethod
    def model_type(self) -> type[_S]:
        raise NotImplementedError()

    @abstractmethod
    def to_model(self, polymorphic: _T) -> _S:
        raise NotImplementedError()

    @abstractmethod
    def from_model(self, model: _S) -> _T:
        raise NotImplementedError()

    def extract_fits_header(self, polymorphic: _T) -> astropy.io.fits.Header | None:
        return None


class NativeAdapter(PolymorphicAdapter[_S, _S]):
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
    def __init__(self) -> None:
        self._validator = pydantic.TypeAdapter(dict[str, AdapterConfig])
        self._config_files_unread = [
            ResourcePath(path) for path in os.environ.get("SHOEFITS_ADAPTER_CONFIGS", "").split(":")
        ]
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
                    factory = AdapterFactory.model_construct(cls=factory)
                self._adapter_factories.setdefault(key, factory)
                if key == tag:
                    adapter = factory.make_adapter()
            if adapter is not None:
                return adapter
        raise KeyError(
            f"No adapter found for polymorphic field with tag {tag} in any of {self._config_files_read}."
        )

    def register_adapter(self, tag: str, adapter: PolymorphicAdapter[Any, Any]) -> None:
        self._adapters[tag] = adapter

    def register_native(self, tag: str, native_type: type[Any]) -> None:
        self._adapters[tag] = NativeAdapter(native_type)


class PolymorphicReadError(RuntimeError):
    pass


class PolymorphicWriteError(RuntimeError):
    pass


class Polymorphic:
    def __init__(self, get_tag: GetPolymorphicTag = get_tag_from_registry):
        self.get_tag = get_tag

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                pcs.dict_schema(pcs.str_schema(), pcs.any_schema()),
                pcs.with_info_plain_validator_function(self.deserialize),
            ]
        )
        # TODO: support generic and/or union source_type.>
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(source_type), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(self.serialize, info_arg=True),
        )

    def deserialize(self, tree: dict[str, JsonValue], info: pydantic.ValidationInfo) -> Any:
        if info.context is None or "polymorphic_adapter_registry" not in info.context:
            raise PolymorphicReadError(
                "Polymorphic field reads require an adapter registry in the Pydantic validation context."
            )
        return self.from_tree(tree, info.context["polymorphic_adapter_registry"])

    def from_tree(self, tree: dict[str, JsonValue], adapter_registry: PolymorphicAdapterRegistry) -> Any:
        match tree:
            case {"tag": str(tag)}:
                pass
            case _:
                raise PolymorphicReadError("No string 'tag' entry found for polymorphic field.")
        adapter: PolymorphicAdapter[Any, Any] = adapter_registry[tag]
        if "tag" not in adapter.model_type.model_fields:
            del tree["tag"]
        serialized = adapter.model_type.model_validate(tree)
        return adapter.from_model(serialized)

    def serialize(
        self,
        obj: Any,
        info: pydantic.SerializationInfo,
    ) -> dict[str, JsonValue]:
        if info.context is None or "polymorphic_adapter_registry" not in info.context:
            raise PolymorphicWriteError(
                "Polymorphic field writes require an adapter registry in the Pydantic serialization context."
            )
        return self.to_tree(
            obj,
            adapter_registry=info.context["polymorphic_adapter_registry"],
            array_writer=info.context.get("array_writer"),
        )

    def to_tree(
        self,
        obj: Any,
        adapter_registry: PolymorphicAdapterRegistry,
        array_writer: asdf_utils.ArrayWriter | None = None,
        header: astropy.io.fits.Header | None = None,
    ) -> dict[str, JsonValue]:
        tag = self.get_tag(obj)
        adapter = adapter_registry[tag]
        serialized = adapter.to_model(obj)
        context: dict[str, Any] = dict(polymorphic_adapter_registry=adapter_registry)
        if array_writer is not None:
            context["array_writer"] = array_writer
        data = serialized.model_dump(context=context)
        if data.setdefault("tag", tag) != tag:
            raise PolymorphicWriteError(
                f"Serialized form already has tag={data['tag']!r}, "
                f"which is inconsistent with tag={tag!r} from the get_tag callback."
            )
        if header is not None and (extracted := adapter.extract_fits_header(obj)):
            header.update(extracted)
        return data
