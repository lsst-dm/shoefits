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

__all__ = ()

from io import BytesIO

import numpy as np

import lsst.shoefits as shf
from lsst.shoefits.examples import afw

adapter_registry = shf.PolymorphicAdapterRegistry()
adapter_registry.register_adapter("astropy_wcs", afw.FitsWcsAdapter())
adapter_registry.register_native("affine_wcs", afw.AffineWcs)
adapter_registry.register_native("chebyshev_bounded_field", afw.ChebyshevBoundedField)


def assert_images_equal(a: shf.Image, b: shf.Image) -> None:
    assert a.bbox == b.bbox
    assert a.unit == b.unit
    np.testing.assert_array_equal(a.array, b.array)


def assert_masks_equal(a: shf.Mask, b: shf.Mask) -> None:
    assert a.bbox == b.bbox
    assert a.schema == b.schema
    np.testing.assert_array_equal(a.array, b.array)


def assert_masked_images_equal(a: afw.MaskedImage, b: afw.MaskedImage) -> None:
    assert_masks_equal(a.mask, b.mask)
    assert_images_equal(a.image, b.image)
    assert_images_equal(a.variance, b.variance)


def test_exposure_round_trip() -> None:
    rng = np.random.RandomState(5)
    bbox = shf.bounds[-3:52, 20:61]
    exposure_in = afw.Exposure.make_example(bbox, rng=rng)
    stream = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(exposure_in, stream, indent=2)
    stream.seek(0)
    reader = shf.FitsReadContext(stream, adapter_registry=adapter_registry)
    exposure_out = reader.read(afw.Exposure)
    assert_masked_images_equal(exposure_in, exposure_out)
