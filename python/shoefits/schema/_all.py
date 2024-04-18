from __future__ import annotations

__all__ = ("SchemaType", "Schema")


from typing import TypeAlias, Union, Annotated

import pydantic

from ._frames import _FrameSchema, FrameSchemaType
from ._simple import _SimpleSchema, _SimpleSchemaType
from ._images import _ImageLikeSchema, _ImageLikeSchemaType
from ._containers import _ContainerSchema, _ContainerSchemaType


SchemaType: TypeAlias = Union[
    _SimpleSchemaType,
    FrameSchemaType,
    _ImageLikeSchemaType,
    _ContainerSchemaType,
]

Schema: TypeAlias = Annotated[
    Union[
        _SimpleSchema,
        _FrameSchema,
        _ImageLikeSchema,
        _ContainerSchema,
    ],
    pydantic.Field(discriminator="schema_type"),
]
