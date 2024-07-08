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

import astropy.wcs
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


def assert_affine_transforms_equal(a: afw.AffineTransform, b: afw.AffineTransform) -> None:
    np.testing.assert_array_equal(a.matrix, b.matrix)


def assert_wcs_equal(a: afw.WcsInterface | None, b: afw.WcsInterface | None) -> None:
    match (a, b):
        case (None, None):
            pass
        case (astropy.wcs.WCS() as a, astropy.wcs.WCS() as b):
            np.testing.assert_array_almost_equal(a.wcs.crpix, b.wcs.crpix)
            np.testing.assert_array_almost_equal(a.wcs.crval, b.wcs.crval)
        case (afw.AffineWcs() as a, afw.AffineWcs() as b):
            assert a.unit == b.unit
            assert_affine_transforms_equal(a.pixel_to_sky, b.pixel_to_sky)
        case _:
            raise AssertionError(f"WCS objects have different types: {type(a).__name__}, {type(b).__name__}.")


def assert_visit_info_equal(a: afw.VisitInfo | None, b: afw.VisitInfo | None) -> None:
    if a is None:
        if b is None:
            return
        else:
            raise AssertionError("First argument is None.")
    elif b is None:
        raise AssertionError("Second argument is None.")
    assert a.exposure_time == b.exposure_time
    assert a.dark_time == b.dark_time
    assert a.mid_time.to_string() == b.mid_time.to_string()


def assert_bounded_fields_equal(a: afw.BoundedField | None, b: afw.BoundedField | None) -> None:
    match (a, b):
        case (None, None):
            pass
        case (afw.ChebyshevBoundedField() as a, afw.ChebyshevBoundedField() as b):
            assert a.bbox == b.bbox
            np.testing.assert_equal(a.coefficients, b.coefficients)
        case _:
            raise AssertionError(f"WCS objects have different types: {type(a).__name__}, {type(b).__name__}.")


def assert_photo_calibs_equal(a: afw.PhotoCalib | None, b: afw.PhotoCalib | None) -> None:
    if a is None:
        if b is None:
            return
        else:
            raise AssertionError("First argument is None.")
    elif b is None:
        raise AssertionError("Second argument is None.")
    assert a.mean == b.mean
    assert a.is_constant == b.is_constant


def assert_exposure_infos_equal(a: afw.ExposureInfo, b: afw.ExposureInfo) -> None:
    assert_wcs_equal(a.wcs, b.wcs)
    assert_visit_info_equal(a.visit_info, b.visit_info)
    assert_photo_calibs_equal(a.photo_calib, b.photo_calib)


def assert_exposures_equal(a: afw.Exposure, b: afw.Exposure) -> None:
    assert_masked_images_equal(a, b)
    assert_exposure_infos_equal(a, b)


def assert_stamp_lists_equal(a: afw.StampList, b: afw.StampList) -> None:
    assert_visit_info_equal(a.visit_info, b.visit_info)
    assert_photo_calibs_equal(a.photo_calib, b.photo_calib)
    assert len(a.stamps) == len(b.stamps)
    for a_stamp, b_stamp in zip(a.stamps, b.stamps):
        assert_wcs_equal(a_stamp.wcs, b_stamp.wcs)
        assert_masked_images_equal(a_stamp, b_stamp)


def test_exposure_round_trip() -> None:
    rng = np.random.RandomState(5)
    bbox = shf.Box.factory[-3:52, 20:61]
    exposure_in = afw.Exposure.make_example(bbox, rng=rng)
    stream = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(exposure_in, stream, indent=2)
    stream.seek(0)
    reader = shf.FitsReadContext(stream, polymorphic_adapter_registry=adapter_registry)
    exposure_out = reader.read(afw.Exposure)
    assert_exposures_equal(exposure_in, exposure_out)


def test_stamp_list_round_trip() -> None:
    rng = np.random.RandomState(5)
    bbox = shf.Box.factory[-3:52, 20:61]
    stamp_list_in = afw.StampList.make_example(bbox, rng=rng)
    stream = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(stamp_list_in, stream, indent=2)
    stream.seek(0)
    reader = shf.FitsReadContext(stream, polymorphic_adapter_registry=adapter_registry)
    stamp_list_out = reader.read(afw.StampList)
    assert_stamp_lists_equal(stamp_list_in, stamp_list_out)
