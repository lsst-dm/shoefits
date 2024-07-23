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

import astropy.io.fits
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

    The model type may be a subclass of `Model` or just `pydantic.BaseModel`.
    If the former, `Model._shoefits_export_fits_header` and
    `Model._shoefits_strip_fits_header` are called in addition to
    `extract_fits_header` and `strip_fits_header` on the adapter (usually
    implementations will implement only one of these pairs, depending on
    whether the original type or the model form is easiest to transform into
    a FITS header).

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

    def extract_fits_header(self, adapted: _T) -> astropy.io.fits.Header | None:
        """Return a FITS header values that should be exported to any HDUs
        associated with the object when it is saved.

        Notes
        -----
        The default implementation returns `None`, resulting in no FITS header
        values being exported.  FITS header exports are stripped on read, not
        read (see `strip_fits_header`).

        See `Model._shoefits_nest` for details of how FITS HDUs are associated
        with objects in a nested data structure.
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

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_ref_schema = pcs.chain_schema(
            [
                self.model_type.__pydantic_core_schema__,
                pcs.no_info_plain_validator_function(self.from_model),
            ]
        )
        if not isinstance(source_type, type):
            raise TypeError(f"Adapters can only be used as an annotation on true types, not {source_type!r}.")
        return pcs.json_or_python_schema(
            json_schema=from_ref_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(source_type), from_ref_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(self.to_model, info_arg=False),
        )
