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
from typing import Annotated, cast

import astropy.io.fits
import astropy.units as u
import numpy as np
import pydantic

import lsst.shoefits as shf

adapter_registry = shf.PolymorphicAdapterRegistry()


def test_image_fits_write() -> None:
    class S(pydantic.BaseModel):
        image: Annotated[shf.Image, shf.FitsOptions(extname="image")]
        alpha: Annotated[float, shf.ExportFitsHeaderKey("ALPHA")]
        beta: int

    s = S(image=shf.Image(0.0, bbox=shf.bounds[1:3, 2:6], unit=u.nJy, dtype=np.int16), alpha=4.125, beta=-2)
    x_grid, y_grid = s.image.bbox.meshgrid
    s.image.array[:, :] = x_grid * 10 + y_grid
    stream = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(s, stream)
    stream.seek(0)
    hdu_list = astropy.io.fits.open(stream)
    stream.seek(0)
    buffer_bytes = stream.getvalue()
    assert len(hdu_list) == 2
    assert [int(k) for k in hdu_list[0].header["SHOEFITS"].split(".")] == list(shf.FORMAT_VERSION)
    assert hdu_list[0].header["ALPHA"] == s.alpha
    primary_bytes = hdu_list[0].data.tobytes()
    tree_size = hdu_list[0].header["TREESIZE"]
    # Check that the tree address and size in the primary header are correct.
    tree_str = primary_bytes[:tree_size].decode("utf-8")
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
            "start": {"x": 2, "y": 1},
        }:
            pass
        case _:
            raise AssertionError(image_tree)
    # Check that the image address in the header is correct.
    assert hdu_list[0].header["LBL00001"] == "image,1"
    image_address = hdu_list[0].header["ADR00001"]
    array = np.frombuffer(
        buffer_bytes[image_address : image_address + s.image.array.size * 2],
        dtype=np.dtype(np.int16).newbyteorder(">"),
    ).reshape(*s.image.array.shape)
    np.testing.assert_array_equal(array, s.image.array)
    # Check that the image HDU has the right data and header.
    assert hdu_list[1].header["EXTNAME"] == "image"
    assert hdu_list[1].header["EXTLEVEL"] == 1
    assert hdu_list[1].header["CRVAL1A"] == 2
    assert hdu_list[1].header["CRVAL2A"] == 1
    np.testing.assert_array_equal(hdu_list[1].data, s.image.array)


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
        mask: Annotated[shf.Mask, shf.FitsOptions(extname=None, mask_header_style=shf.MaskHeaderFormat.AFW)]

    s = S(mask=shf.Mask(bbox=shf.bounds[1:5, -2:6], schema=mask_schema))
    s.mask.array[0, 0, :] = mask_schema.bitmask("bad", "interpolated")
    s.mask.array[0, 2, :] = mask_schema.bitmask("saturated")
    s.mask.array[1, 3, :] = mask_schema.bitmask("interpolated", "fill5")
    buffer = BytesIO()
    shf.FitsWriteContext(adapter_registry).write(s, buffer)
    buffer.seek(0)
    hdu_list = astropy.io.fits.open(buffer)
    buffer.seek(0)
    buffer_bytes = buffer.getvalue()
    assert len(hdu_list) == 2
    assert [int(k) for k in hdu_list[0].header["SHOEFITS"].split(".")] == list(shf.FORMAT_VERSION)
    assert "MP_BAD" not in hdu_list[0].header
    primary_bytes = hdu_list[0].data.tobytes()
    tree_size = hdu_list[0].header["TREESIZE"]
    # Check that the tree address and size in the primary header are correct.
    tree_str = primary_bytes[:tree_size].decode("utf-8")
    tree: shf.json_utils.JsonValue = json.loads(tree_str)
    match tree:
        case {"mask": mask_tree}:
            pass
        case _:
            raise AssertionError(tree)
    match mask_tree:
        case {
            "data": {
                "source": "fits:2",
                "shape": [4, 8, 2],
                "datatype": "uint8",
                "byteorder": "big",
            },
            "start": {"x": -2, "y": 1},
            "planes": list(planes_list),
        }:
            pass
        case _:
            raise AssertionError(mask_tree)
    assert len(planes_list) == len(mask_schema)
    assert shf.MaskPlane(**cast(dict, planes_list[0])) == mask_schema[0]
    assert planes_list[1] is None
    # Check that the image address in the tree is correct.
    assert hdu_list[0].header["LBL00001"] == 2
    image_address = hdu_list[0].header["ADR00001"]
    array = np.frombuffer(
        buffer_bytes[image_address : image_address + s.mask.array.size],
        dtype=np.dtype(np.uint8).newbyteorder(">"),
    ).reshape(*s.mask.array.shape)
    np.testing.assert_array_equal(array, s.mask.array)
    # Check that the image HDU has the right data and header.
    assert hdu_list[1].header["EXTLEVEL"] == 1
    assert hdu_list[1].header["CRVAL1A"] == -2
    assert hdu_list[1].header["CRVAL2A"] == 1
    assert hdu_list[1].header["MP_BAD"] == 0
    assert hdu_list[1].header["MP_INTERPOLATED"] == 2
    np.testing.assert_array_equal(hdu_list[1].data, s.mask.array)
