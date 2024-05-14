from __future__ import annotations

__all__ = (
    "YamlModel",
    "RestrictedYamlLoader",
    "DeferredYaml",
    "YamlValue",
    "yaml_represent_mapping",
    "yaml_represent_str",
)

from collections.abc import Callable
from typing import Any, ClassVar, TypeAlias, Union, cast

import pydantic
import yaml


class RestrictedYamlLoader(yaml.SafeLoader):
    pass


class DeferredYaml:
    def __init__(self, callback: Callable[[yaml.Dumper, Any, str], yaml.Node], data: Any, tag: str):
        self.callback = callback
        self.data = data
        self.tag = tag


def _yaml_model_represent(dumper: yaml.Dumper, obj: DeferredYaml) -> yaml.Node:
    return obj.callback(dumper, obj.data, obj.tag)


yaml.add_representer(DeferredYaml, _yaml_model_represent)


class YamlModel(pydantic.BaseModel):
    def __init_subclass__(cls, yaml_tag: str | None = None, **kwargs: Any):
        if yaml_tag is not None:
            cls.yaml_tag = yaml_tag
            RestrictedYamlLoader.add_constructor(yaml_tag, cls._construct_yaml)
            # TODO: add 'tag' to JSON schema, nondestructively
        return super().__init_subclass__(**kwargs)

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    yaml_tag: ClassVar[str]

    @pydantic.model_serializer(mode="wrap")
    def _yaml_model_serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> DeferredYaml | dict[str, Any]:
        result = self._serialize(handler, info)
        if info.context is not None and info.context.get("yaml"):
            return DeferredYaml(yaml_represent_mapping, result, self.yaml_tag)
        return result

    def _serialize(
        self, handler: pydantic.SerializerFunctionWrapHandler, info: pydantic.SerializationInfo
    ) -> dict[str, Any]:
        # This is intended for subclasses to override, since they can't further
        # customize the serializer via Pydantic methods.
        return handler(self)

    @classmethod
    def _construct_yaml(
        cls, loader: yaml.Loader | yaml.FullLoader | yaml.SafeLoader, node: yaml.Node
    ) -> YamlModel:
        return cls.model_validate(loader.construct_mapping(cast(yaml.MappingNode, node)))


YamlValue: TypeAlias = Union[int, str, float, DeferredYaml, None, list["YamlValue"], dict[str, "YamlValue"]]


def yaml_represent_mapping(dumper: yaml.Dumper, data: dict[str, YamlValue], tag: str) -> yaml.MappingNode:
    return dumper.represent_mapping(tag, data)


def yaml_represent_str(dumper: yaml.Dumper, data: str, tag: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(tag, data)
