from __future__ import annotations

__all__ = ()

import json
from io import BytesIO

import astropy.io.fits
import astropy.units as u
import numpy as np
import shoefits as shf


def test_image_fits_write() -> None:
    class S(shf.Struct):
        def __init__(self) -> None:
            super().__init__()
            self.image = shf.Image.from_zeros(np.int16, shf.bounds[1:3, 2:6], u.Unit("Jy"))
            self.alpha = 4.125
            self.beta = -2
            x_grid, y_grid = self.image.bbox.meshgrid
            self.image.array[:, :] = x_grid * 10 + y_grid

        image: shf.Image = shf.Field(dtype=np.int16)
        alpha: float = shf.Field(fits_header=True)
        beta: int = shf.Field(fits_header=False)

    s = S()
    writer = shf.FitsWriter(s)
    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    hdu_list = astropy.io.fits.open(buffer)
    buffer.seek(0)
    buffer_bytes = buffer.getvalue()
    assert len(hdu_list) == 3
    assert not hdu_list[0].data
    assert [int(k) for k in hdu_list[0].header["SHOEFITS"].split(".")] == list(shf.FORMAT_VERSION)
    assert hdu_list[0].header["ALPHA"] == s.alpha
    # Check that the tree address and size in the primary header are correct.
    tree_address = hdu_list[0].header["TREEADDR"]
    tree_size = hdu_list[0].header["TREESIZE"]
    tree_str = buffer_bytes[tree_address : tree_address + tree_size].decode("utf-8")
    tree = json.loads(tree_str)
    match tree:
        case {"image": image_tree, "alpha": s.alpha, "beta": s.beta}:
            pass
        case _:
            raise AssertionError(tree)
    match image_tree:
        case {
            "data": {
                "value": {
                    "source": "fits:image",
                    "shape": [2, 4],
                    "datatype": "int16",
                    "byteorder": "big",
                },
                "unit": "Jy",
            },
            "start": {"x": 2, "y": 1},
            "address": int(image_address),
        }:
            pass
        case _:
            raise AssertionError(image_tree)
    # Check that the image address in the tree is correct.
    array = np.frombuffer(
        buffer_bytes[image_address : image_address + s.image.array.size * 2],
        dtype=np.dtype(np.int16).newbyteorder(">"),
    ).reshape(2, 4)
    np.testing.assert_array_equal(array, s.image.array)
    # Check that the image HDU has the right data and header.
    assert hdu_list[1].header["EXTNAME"] == "image"
    assert hdu_list[1].header["EXTLEVEL"] == 1
    assert hdu_list[1].header["CRVAL1A"] == 2
    assert hdu_list[1].header["CRVAL2A"] == 1
    assert hdu_list[1].header["ALPHA"] == s.alpha
    np.testing.assert_array_equal(hdu_list[1].data, s.image.array)
    # Check that the tree is also readable via FITS, and has the right header.
    tree_str_fits = hdu_list[-1].data.tobytes().decode("utf-8")
    assert tree_str_fits == tree_str
    assert hdu_list[-1].header["EXTNAME"] == "JSONTREE"


def test_compressed_mask_fits_write() -> None:
    mask_schema = shf.MaskSchema(
        [
            shf.MaskPlane("bad", "Physical pixel is unreliable."),
            None,
            shf.MaskPlane("interpolated", "Pixel value was interpolated."),
            shf.MaskPlane("saturated", "Pixel was saturated."),
        ]
        + [shf.MaskPlane(f"fill{d}", "Filler mask plane") for d in range(6)]
    )

    class S(shf.Struct):
        def __init__(self) -> None:
            super().__init__()
            self.mask = shf.Mask.from_zeros(np.uint8, shf.bounds[1:3, 2:6], mask_schema)
            self.mask.array[0, 0, :] = mask_schema.bitmask("bad", "interpolated")
            self.mask.array[0, 2, :] = mask_schema.bitmask("saturated")
            self.mask.array[1, 3, :] = mask_schema.bitmask("interpolated", "fill5")

        mask: shf.Mask = shf.Field()

    s = S()
    writer = shf.FitsWriter(s)
    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    hdu_list = astropy.io.fits.open(buffer)
    buffer.seek(0)
    buffer_bytes = buffer.getvalue()
    assert len(hdu_list) == 3
    assert not hdu_list[0].data
    assert [int(k) for k in hdu_list[0].header["SHOEFITS"].split(".")] == list(shf.FORMAT_VERSION)
    assert "MP_BAD" not in hdu_list[0].header
    # Check that the tree address and size in the primary header are correct.
    tree_address = hdu_list[0].header["TREEADDR"]
    tree_size = hdu_list[0].header["TREESIZE"]
    tree_str = buffer_bytes[tree_address : tree_address + tree_size].decode("utf-8")
    tree = json.loads(tree_str)
    match tree:
        case {"mask": mask_tree}:
            pass
        case _:
            raise AssertionError(tree)
    match mask_tree:
        case {
            "data": {
                "source": "fits:mask",
                "shape": [2, 4, 2],
                "datatype": "uint8",
                "byteorder": "big",
            },
            "start": {"x": 2, "y": 1},
            "address": int(image_address),
            "planes": planes_list,
        }:
            pass
        case _:
            raise AssertionError(mask_tree)
    assert len(planes_list) == len(mask_schema)
    assert shf.MaskPlane(**planes_list[0]) == mask_schema[0]
    assert planes_list[1] is None
    # Check that the image address in the tree is correct.
    array = np.frombuffer(
        buffer_bytes[image_address : image_address + s.mask.array.size],
        dtype=np.dtype(np.uint8).newbyteorder(">"),
    ).reshape(2, 4, 2)
    np.testing.assert_array_equal(array, s.mask.array)
    # Check that the image HDU has the right data and header.
    assert hdu_list[1].header["EXTNAME"] == "mask"
    assert hdu_list[1].header["EXTLEVEL"] == 1
    assert hdu_list[1].header["CRVAL1A"] == 2
    assert hdu_list[1].header["CRVAL2A"] == 1
    assert hdu_list[1].header["MP_BAD"] == 0
    assert hdu_list[1].header["MP_INTERPOLATED"] == 2
    np.testing.assert_array_equal(hdu_list[1].data, s.mask.array)
    # Check that the tree is also readable via FITS, and has the right header.
    tree_str_fits = hdu_list[-1].data.tobytes().decode("utf-8")
    assert tree_str_fits == tree_str
    assert hdu_list[-1].header["EXTNAME"] == "JSONTREE"
