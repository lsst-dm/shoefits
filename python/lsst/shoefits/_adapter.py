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

__all__ = ("Adapter",)

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import pydantic
import pydantic_core.core_schema as pcs

_T = TypeVar("_T")
_S = TypeVar("_S", bound=pydantic.BaseModel)


class Adapter(Generic[_T, _S]):
    """A helper class that transforms arbitrary objects to Pydantic models for
    serialization (and back).

    Notes
    -----
    `Adapter` has two generic type parameters:

    - the arbitrary type being adapted;
    - the Pydantic model type it is mapped to for serialization (also exposed
      at runtime via the `model_type` property).

    Instances of subclass implementations of `Adapter` can be used directly
    with `typing.Annotated` to implement proxied serialization and validation
    (i.e. reading) for an arbitrary type::

        from typing import Annotated, TypeAlias

        class Thing:

            def __init__(self, value: int):
                self.value = value

        class ThingModel(pydantic.BaseModel):
            value: int

        class ThingAdapter(Adapter[Thing, ThingModel]):
            @property
            def model_type(self) -> type[ThingModel]:
                return ThingModel
            def to_model(self, thing: Thing) -> ThingModel:
                return ThingModel(value=thing.value)
            def from_model(self, model: ThingModel) -> Thing:
                return Thing(model.value)

        SerializableThing: TypeAlias = Annotated[Thing, ThingAdapter]

    Adapters are also used by the `Polymorphic` annotation class to support
    cases where the set of possible runtime types is not known in advance.

    The model type should usually not be a subclass of the `Model` intermediate
    base class, because `Model` and `Adapter` both support frame nesting
    and header updates, and these interact in usually-undesirable ways.
    """

    @property
    @abstractmethod
    def model_type(self) -> type[_S]:
        """The Pydantic model type the adapted type is mapped to for
        serialization.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_model(self, polymorphic: _T, /) -> _S:
        """Convert an adapted object to a Pydantic model instance."""
        raise NotImplementedError()

    @abstractmethod
    def from_model(self, model: _S, /) -> _T:
        """Construct an adapted object from a Pydantic model instance."""
        raise NotImplementedError()

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                self.model_type.__pydantic_core_schema__,
                pcs.no_info_plain_validator_function(self.from_model),
            ]
        )
        if not isinstance(source_type, type):
            raise TypeError(f"Adapters can only be used as an annotation on true types, not {source_type!r}.")
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(source_type), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(self.to_model, info_arg=False),
        )
