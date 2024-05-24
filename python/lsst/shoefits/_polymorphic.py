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
    "register_tag",
    "get_tag_from_registry",
)

import os
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar, final, overload

import pydantic

from lsst.resources import ResourcePath
from lsst.utils.doImport import doImportType

if TYPE_CHECKING:
    from ._struct import Struct

_C = TypeVar("_C", bound=type[Any])
_T = TypeVar("_T")
_S = TypeVar("_S", bound="Struct")


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
    def struct_type(self) -> type[_S]:
        raise NotImplementedError()

    @abstractmethod
    def to_struct(self, polymorphic: _T) -> _S:
        raise NotImplementedError()

    @abstractmethod
    def from_struct(self, struct: _S) -> _T:
        raise NotImplementedError()


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
        self._adapters: dict[str, PolymorphicAdapter[Any, Any]] = {}

    def __getitem__(self, tag: str) -> PolymorphicAdapter[Any, Any]:
        adapter: PolymorphicAdapter[Any, Any] | None = None
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
