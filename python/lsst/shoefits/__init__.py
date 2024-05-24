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

__all__ = (
    "__dependency_versions__",
    "__fingerprint__",
    "__repo_version__",
    "__version__",
    "bounds",
    "Box",
    "Extent",
    "field",
    "Field",
    "FieldInfo",
    "FitsCompression",
    "FitsCompressionAlgorithm",
    "FitsWriter",
    "FORMAT_VERSION",
    "Frame",
    "GetPolymorphicTag",
    "HeaderFieldInfo",
    "Image",
    "ImageFieldInfo",
    "Interval",
    "MappingFieldInfo",
    "Mask",
    "MaskFieldInfo",
    "MaskPlane",
    "MaskSchema",
    "ModelFieldInfo",
    "Point",
    "PolymorphicAdapter",
    "PolymorphicAdapterRegistry",
    "SequenceFieldInfo",
    "Struct",
    "StructFieldInfo",
    "ValueFieldInfo",
    "asdf_utils",
    "json_utils",
)

from . import asdf_utils, json_utils
from ._compression import FitsCompression, FitsCompressionAlgorithm
from ._field_info import (
    FieldInfo,
    HeaderFieldInfo,
    ImageFieldInfo,
    MappingFieldInfo,
    MaskFieldInfo,
    ModelFieldInfo,
    SequenceFieldInfo,
    StructFieldInfo,
    ValueFieldInfo,
)
from ._fits_writer import FORMAT_VERSION, FitsWriter
from ._frame import Frame
from ._geom import Box, Extent, Interval, Point, bounds
from ._image import Image
from ._mask import Mask, MaskPlane, MaskSchema
from ._polymorphic import GetPolymorphicTag, PolymorphicAdapter, PolymorphicAdapterRegistry
from ._struct import Field, Struct, field
from .version import __dependency_versions__, __fingerprint__, __repo_version__, __version__
