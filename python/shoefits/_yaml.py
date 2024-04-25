from __future__ import annotations

__all__ = ("YamlModel", "RestrictedYamlLoader", "DeferredYaml")

from collections.abc import Callable
from typing import ClassVar, Any, cast

import yaml

import pydantic


class RestrictedYamlLoader(yaml.SafeLoader):
    pass


class DeferredYaml:
    def __init__(self, callback: Callable[[yaml.Dumper, Any], yaml.Node], data: Any, extra: Any = None):
        self.callback = callback
        self.data = data
        self.extra = extra


def _yaml_model_represent(dumper: yaml.Dumper, obj: DeferredYaml) -> yaml.Node:
    return obj.callback(dumper, obj.data)


yaml.add_representer(DeferredYaml, _yaml_model_represent)


class YamlModel(pydantic.BaseModel):
    def __init_subclass__(cls, yaml_tag: str | None = None, **kwargs: Any):
        if yaml_tag is not None:
            cls.yaml_tag = yaml_tag
            RestrictedYamlLoader.add_constructor(yaml_tag, cls._construct_yaml)
        return super().__init_subclass__(**kwargs)

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    yaml_tag: ClassVar[str]

    @pydantic.model_serializer(mode="wrap")
    def _serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> DeferredYaml | dict[str, Any]:
        result = handler(self)
        if info.context is not None and info.context.get("yaml"):
            return DeferredYaml(self._represent_yaml, result, self)
        return result

    @classmethod
    def _construct_yaml(
        cls, loader: yaml.Loader | yaml.FullLoader | yaml.SafeLoader, node: yaml.Node
    ) -> YamlModel:
        return cls.model_validate(loader.construct_mapping(cast(yaml.MappingNode, node)))

    @classmethod
    def _represent_yaml(cls, dumper: yaml.Dumper, data: dict[str, Any]) -> yaml.MappingNode:
        return dumper.represent_mapping(cls.yaml_tag, data)
