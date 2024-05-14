from __future__ import annotations

__all__ = ()

from io import BytesIO

import astropy.io.fits
import astropy.units as u
import numpy as np
import shoefits as shf


def test_image_fits_write():
    class S(shf.Struct):
        def __init__(self) -> None:
            super().__init__()
            self.image = shf.Image.from_zeros(np.float32, shf.bounds[1:3, 2:6], u.Unit("Jy"))
            self.alpha_x = 4.125
            self.alpha_y = 1.50
            self.beta = -2.25
            x_grid, y_grid = self.image.bbox.meshgrid
            self.image.array[:, :] = self.alpha_x * x_grid + self.alpha_y * y_grid + self.beta

        image: shf.Image = shf.Field(dtype=np.float32)
        alpha_x: float = shf.Field(fits_header=True)
        alpha_y: float = shf.Field(fits_header="ALPHY")
        beta: float = shf.Field(fits_header=False)

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
    tree_address = hdu_list[0].header["TREEADDR"]
    tree_size = hdu_list[0].header["TREESIZE"]
    tree_str = buffer_bytes[tree_address : tree_address + tree_size].decode("utf-8")
    assert tree_str.startswith("#ASDF 1.0.0")
    assert tree_str.endswith("...\n")
