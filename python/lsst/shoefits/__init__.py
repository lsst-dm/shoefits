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
    "FitsCompression",
    "FitsCompressionAlgorithm",
    "FitsWriteContext",
    "FitsDataOptions",
    "FORMAT_VERSION",
    "GetPolymorphicTag",
    "Image",
    "ImageReference",
    "Interval",
    "Mask",
    "MaskPlane",
    "MaskReference",
    "MaskSchema",
    "Point",
    "Polymorphic",
    "PolymorphicAdapter",
    "PolymorphicAdapterRegistry",
    "PolymorphicReadError",
    "PolymorphicWriteError",
    "ReadContext",
    "ReadError",
    "WriteContext",
    "WriteError",
    "register_tag",
    "keywords",
    "asdf_utils",
    "json_utils",
)

from . import asdf_utils, json_utils, keywords
from ._fits_options import FitsCompression, FitsCompressionAlgorithm, FitsDataOptions
from ._fits_write_context import FORMAT_VERSION, FitsWriteContext
from ._geom import Box, Extent, Interval, Point, bounds
from ._image import Image, ImageReference
from ._mask import Mask, MaskPlane, MaskReference, MaskSchema
from ._polymorphic import (
    GetPolymorphicTag,
    Polymorphic,
    PolymorphicAdapter,
    PolymorphicAdapterRegistry,
    PolymorphicReadError,
    PolymorphicWriteError,
    register_tag,
)
from ._read_context import ReadContext, ReadError
from ._write_context import WriteContext, WriteError
from .version import __dependency_versions__, __fingerprint__, __repo_version__, __version__
