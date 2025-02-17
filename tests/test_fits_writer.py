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

import json
from io import BytesIO
from typing import Annotated

import astropy.io.fits
import astropy.units as u
import numpy as np
import pydantic

import lsst.shoefits as shf

adapter_registry = shf.PolymorphicAdapterRegistry()


def test_image_fits_write() -> None:
    class S(pydantic.BaseModel):
        image: Annotated[shf.Image, shf.Fits(extname="image")]
        alpha: Annotated[float, shf.ExportFitsHeaderKey("ALPHA")]
        beta: int

    s = S(
        image=shf.Image(0.0, bbox=shf.Box.factory[1:3, 2:6], unit=u.nJy, dtype=np.int16),
        alpha=4.125,
        beta=-2,
    )
    x_grid, y_grid = np.meshgrid(s.image.bbox.x.arange, s.image.bbox.y.arange)
    s.image.array[:, :] = x_grid * 10 + y_grid
    stream = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(s, stream)
    stream.seek(0)
    hdu_list = astropy.io.fits.open(stream, lazy_load_hdus=False)
    assert len(hdu_list) == 3
    assert [int(k) for k in hdu_list[0].header["SHOEFITS"].split(".")] == list(shf.FORMAT_VERSION)
    assert hdu_list[0].header["ALPHA"] == s.alpha
    # Check that the image HDU has the right data and header.
    assert hdu_list[1].header["EXTNAME"] == "image"
    assert hdu_list[1].header["CRVAL1A"] == s.image.bbox.x.start
    assert hdu_list[1].header["CRVAL2A"] == s.image.bbox.y.start
    np.testing.assert_array_equal(hdu_list[1].data, s.image.array)
    # Check that the JSON/tree HDU is what we expect.
    assert hdu_list[2].header["EXTNAME"] == "tree"
    tree_bytes = hdu_list[2].data[0]["json"].tobytes()
    tree_str = tree_bytes.decode()
    tree: shf.json_utils.JsonValue = json.loads(tree_str)
    match tree:
        case {"image": image_tree, "alpha": s.alpha, "beta": s.beta}:
            pass
        case _:
            raise AssertionError(tree)
    match image_tree:
        case {
            "data": {
                "value": {
                    "source": "fits:image,1",
                    "shape": [2, 4],
                    "datatype": "int16",
                    "byteorder": "big",
                },
                "unit": "nJy",
            },
            "start": [1, 2],
        }:
            pass
        case _:
            raise AssertionError(image_tree)
    # Test that the addresses we've stuffed into the headers are correct.
    tree_address = hdu_list[0].header[shf.keywords.TREE_ADDRESS]
    assert tree_address == hdu_list[0].filebytes() + hdu_list[1].filebytes()
    assert hdu_list[2].header["LBL00001"] == "image,1"
    image_address = hdu_list[2].header["ADR00001"]
    assert image_address == hdu_list[0].filebytes() + len(hdu_list[1].header.tostring())


def test_mask_fits_write() -> None:
    mask_schema = shf.MaskSchema(
        [
            shf.MaskPlane("bad", "Physical pixel is unreliable."),
            None,
            shf.MaskPlane("interpolated", "Pixel value was interpolated."),
            shf.MaskPlane("saturated", "Pixel was saturated."),
        ]
        + [shf.MaskPlane(f"fill{d}", "Filler mask plane") for d in range(6)]
    )

    class S(pydantic.BaseModel):
        mask: Annotated[shf.Mask, shf.Fits(extname="mask", mask_header_style="afw")]

    s = S(mask=shf.Mask(bbox=shf.Box.factory[1:5, -2:6], schema=mask_schema))
    s.mask.array[0, 0, :] = mask_schema.bitmask("bad", "interpolated")
    s.mask.array[0, 2, :] = mask_schema.bitmask("saturated")
    s.mask.array[1, 3, :] = mask_schema.bitmask("interpolated", "fill5")
    buffer = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(s, buffer)
    buffer.seek(0)
    hdu_list = astropy.io.fits.open(buffer)
    assert len(hdu_list) == 3
    assert [int(k) for k in hdu_list[0].header["SHOEFITS"].split(".")] == list(shf.FORMAT_VERSION)
    assert "MP_BAD" not in hdu_list[0].header
    # Check that the mask HDU has the right data and header.
    assert hdu_list[1].header["EXTNAME"] == "mask"
    assert hdu_list[1].header["NAXIS1"] == s.mask.schema.mask_size
    assert hdu_list[1].header["NAXIS2"] == s.mask.bbox.x.size
    assert hdu_list[1].header["NAXIS3"] == s.mask.bbox.y.size
    assert hdu_list[1].header["CRVAL2A"] == s.mask.bbox.x.start
    assert hdu_list[1].header["CRVAL3A"] == s.mask.bbox.y.start
    assert hdu_list[1].header["MP_BAD"] == 0
    assert hdu_list[1].header["MP_INTERPOLATED"] == 2
    np.testing.assert_array_equal(hdu_list[1].data, s.mask.array)
    # Check that the JSON is what we expect.
    assert hdu_list[2].header["EXTNAME"] == "tree"
    tree_bytes = hdu_list[2].data[0]["json"].tobytes()
    tree_str = tree_bytes.decode()
    tree: shf.json_utils.JsonValue = json.loads(tree_str)
    match tree:
        case {"mask": mask_tree}:
            pass
        case _:
            raise AssertionError(tree)
    match mask_tree:
        case {
            "data": {
                "source": "fits:mask,1",
                "shape": [4, 8, 2],
                "datatype": "uint8",
                "byteorder": "big",
            },
            "start": [1, -2],
            "planes": list(),
        }:
            pass
        case _:
            raise AssertionError(mask_tree)
    # Check that the tree address and size in the primary header are correct.
    # Test that the addresses we've stuffed into the headers are correct.
    tree_address = hdu_list[0].header[shf.keywords.TREE_ADDRESS]
    assert tree_address == hdu_list[0].filebytes() + hdu_list[1].filebytes()
    assert hdu_list[2].header["LBL00001"] == "mask,1"
    image_address = hdu_list[2].header["ADR00001"]
    assert image_address == hdu_list[0].filebytes() + len(hdu_list[1].header.tostring())
