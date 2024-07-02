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

from abc import ABC, abstractmethod
from typing import Annotated, Any, Protocol, Self, overload

import astropy.io.fits
import astropy.time
import astropy.units
import astropy.wcs.wcsapi
import numpy as np
import numpy.typing as npt
import pydantic
from astropy.coordinates import SkyCoord

import lsst.shoefits as shf


class AffineTransform(pydantic.BaseModel):
    xx: float = 1.0
    xy: float = 0.0
    yx: float = 0.0
    yy: float = 1.0
    x: float = 0.0
    y: float = 0.0

    @overload
    def __call__(self, x: float, y: float) -> tuple[float, float]: ...

    @overload
    def __call__(self, x: np.ndarray | float, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def __call__(self, x: np.ndarray, y: float) -> tuple[np.ndarray, np.ndarray]: ...

    def __call__(self, x: Any, y: Any) -> tuple[Any, Any]:
        return (self.yx * x + self.yy * y + self.y, self.xx * x + self.xy * y + self.x)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.xx, self.xy, self.x],
                [self.yx, self.yy, self.y],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def determinant(self) -> float:
        return self.xx * self.yy - self.xy * self.yx

    def inverted(self) -> AffineTransform:
        (
            (xx, xy, x),
            (yx, yy, y),
            _,
        ) = np.linalg.inv(self.matrix)
        return AffineTransform(xx=xx, xy=xy, yx=yx, yy=yy, x=x, y=y)

    def then(self, other: AffineTransform) -> AffineTransform:
        (
            (xx, xy, x),
            (yx, yy, y),
            _,
        ) = np.dot(other.matrix, self.matrix)
        return AffineTransform(xx=xx, xy=xy, yx=yx, yy=yy, x=x, y=y)


class BoundedField(pydantic.BaseModel, ABC):
    bbox: shf.Box  # domain of bounded field

    @overload
    def evaluate(self, x: float, y: float) -> float: ...

    @overload
    def evaluate(self, x: np.ndarray | float, y: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, x: np.ndarray, y: float) -> np.ndarray: ...

    @abstractmethod
    def evaluate(self, x: Any, y: Any) -> Any:
        raise NotImplementedError()


@shf.register_tag("chebyshev_bounded_field")
class ChebyshevBoundedField(BoundedField):
    to_chebyshev_domain: AffineTransform  # maps bbox to [-1,1]x[-1,1]
    coefficients: shf.Array  # shape=(order_x + 1, order_y + 1)

    @overload
    def evaluate(self, x: float, y: float) -> float: ...

    @overload
    def evaluate(self, x: np.ndarray | float, y: np.ndarray) -> np.ndarray: ...

    @overload
    def evaluate(self, x: np.ndarray, y: float) -> np.ndarray: ...

    def evaluate(self, x: Any, y: Any) -> Any:
        u, v = self.to_chebyshev_domain(x, y)
        return np.polynomial.chebyshev.chebval2d(self.coefficients, u, v)

    @classmethod
    def make_linear(cls, bbox: shf.Box, z: float, dz_dx: float, dz_dy: float) -> ChebyshevBoundedField:
        to_chebyshev_domain = AffineTransform(
            xx=2.0 / bbox.size.x,
            yy=2.0 / bbox.size.y,
            x=-bbox.x.center,
            y=-bbox.y.center,
        )
        coefficients = np.array([[z, dz_dx], [dz_dy, 0.0]], dtype=float)
        return cls(bbox=bbox, to_chebyshev_domain=to_chebyshev_domain, coefficients=coefficients)


class PhotoCalib(pydantic.BaseModel):
    mean: float
    is_constant: bool
    calibration: Annotated[BoundedField, shf.Polymorphic()]

    @classmethod
    def make_example(cls, bbox: shf.Box, rng: np.random.RandomState) -> PhotoCalib:
        calibration = ChebyshevBoundedField.make_linear(
            bbox,
            z=rng.uniform(0.9, 1.0),
            dz_dx=rng.uniform(-0.05, 0.05),
            dz_dy=rng.uniform(-0.05, 0.05),
        )
        return cls(mean=0.4, is_constant=False, calibration=calibration)


class WcsInterface(Protocol):
    def pixel_to_world(self, x: float, y: float) -> SkyCoord: ...

    def world_to_pixel(self, sky: SkyCoord) -> tuple[float, float]: ...


def get_wcs_tag(wcs: WcsInterface) -> str:
    match wcs:
        case astropy.wcs.WCS():
            return "astropy_wcs"
        case AffineWcs():
            return "affine_wcs"
    raise LookupError(type(wcs).__name__)


class FitsWcsModel(pydantic.BaseModel):
    fits: dict[str, float | int | str]


class FitsWcsAdapter(shf.PolymorphicAdapter[astropy.wcs.WCS, FitsWcsModel]):
    @property
    def model_type(self) -> type[FitsWcsModel]:
        return FitsWcsModel

    def to_model(self, polymorphic: astropy.wcs.WCS) -> FitsWcsModel:
        header: astropy.io.fits.Header = polymorphic.to_header()
        return FitsWcsModel(fits=dict(header))

    def from_model(self, model: FitsWcsModel) -> astropy.wcs.WCS:
        header = astropy.io.fits.Header()
        header.update(model.fits)
        return astropy.wcs.WCS(header)

    def extract_fits_header(self, polymorphic: astropy.wcs.WCS) -> astropy.io.fits.Header:
        return polymorphic.to_header()


def make_example_fits_wcs(bbox: shf.Box, rng: np.random.RandomState) -> astropy.wcs.WCS:
    return astropy.wcs.WCS(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 32.5 + rng.uniform(-1.0, 1.0),
            "CRVAL2": 11.2 + rng.uniform(-1.0, 1.0),
            "CRPIX1": bbox.x.center,
            "CRPIX2": bbox.y.center,
            "CD1_1": 5e-5,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": -5e-5,
            "RADESYS": "ICRS",
        }
    )


class AffineWcs(pydantic.BaseModel):
    pixel_to_sky: AffineTransform
    unit: shf.Unit = astropy.units.degree

    def pixel_to_world(self, x: float, y: float) -> SkyCoord:
        ra, dec = self.pixel_to_sky(x, y)
        return SkyCoord(
            ra=ra * self.unit,
            dec=dec * self.unit,
        )

    def world_to_pixel(self, sky: SkyCoord) -> tuple[float, float]:
        assert sky.frame.name == "icrs"
        return self.pixel_to_sky.inverted()(sky.ra.to_value(self.unit), sky.dec.to_value(self.unit))

    @classmethod
    def approximate(
        cls,
        other: astropy.wcs.WCS,
        x: float,
        y: float,
        epsilon: float = 0.01,
    ) -> Self:
        x_inputs = np.array([x, x - epsilon, x + epsilon, x, x], dtype=float)
        y_inputs = np.array([y, y, y, y - epsilon, y + epsilon], dtype=float)
        h = 2 * epsilon
        sky: SkyCoord = other.pixel_to_world(x_inputs, y_inputs).transform_to("icrs")
        ra = sky.ra.to_value(astropy.units.degree)
        dec = sky.dec.to_value(astropy.units.degree)
        pixel_to_sky = AffineTransform(x=-x, y=-y).then(
            AffineTransform(
                xx=(ra[2] - ra[1]) / h,
                xy=(ra[4] - ra[3]) / h,
                yx=(dec[2] - dec[1]) / h,
                yy=(dec[4] - dec[3]) / h,
                x=ra[0],
                y=dec[0],
            )
        )
        return cls(pixel_to_sky=pixel_to_sky, unit=astropy.units.degree)


class VisitInfo(pydantic.BaseModel):
    exposure_time: shf.Quantity
    dark_time: shf.Quantity
    mid_time: shf.Time
    instrument: str
    id: int
    observation_type: str
    science_program: str

    @shf.fits_header_exporter
    def to_header(self) -> astropy.io.fits.Header:
        header = astropy.io.fits.Header()
        header["EXPTIME"] = self.exposure_time.to_value(astropy.units.s)
        return header

    @classmethod
    def make_example(cls, rng: np.random.RandomState) -> Self:
        return cls(
            exposure_time=30.0 * astropy.units.s,
            dark_time=0.0 * astropy.units.s,
            mid_time=(
                astropy.time.Time("2024-06-03 17:19:21.156", scale="tai", format="iso")
                + rng.uniform(35.0, 60.0) * astropy.units.s
            ),
            instrument="ImaginaryCam",
            id=47,
            observation_type="science",
            science_program="ultra mega deep",
        )


class ExposureInfo(pydantic.BaseModel):
    wcs: Annotated[WcsInterface, shf.Polymorphic(get_wcs_tag)] | None = None
    visit_info: VisitInfo | None = None
    photo_calib: PhotoCalib | None = None


class MaskedImage(pydantic.BaseModel):
    image: Annotated[shf.Image, shf.FitsOptions(extname="image")] = pydantic.Field(frozen=True)
    mask: Annotated[shf.Mask, shf.FitsOptions(extname="mask", mask_header_style=shf.MaskHeaderStyle.AFW)] = (
        pydantic.Field(frozen=True)
    )
    variance: Annotated[shf.Image, shf.FitsOptions(extname="variance")] = pydantic.Field(frozen=True)

    @classmethod
    def from_image(
        cls, image: shf.Image, mask_schema: shf.MaskSchema, variance: float = 0.0, **kwargs: Any
    ) -> Self:
        return cls.model_construct(
            image=image,
            mask=shf.Mask(0, bbox=image.bbox, schema=mask_schema),
            variance=shf.Image(
                variance,
                bbox=image.bbox,
                unit=(image.unit**2 if image.unit is not None else None),
                dtype=np.float32,
            ),
            **kwargs,
        )

    @classmethod
    def from_bbox(
        cls,
        bbox: shf.Box,
        mask_schema: shf.MaskSchema,
        fill: float = 0,
        variance: float = 0.0,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> Self:
        return cls.from_image(shf.Image(fill, dtype=dtype, bbox=bbox), mask_schema, variance, **kwargs)

    @classmethod
    def from_section(cls, other: MaskedImage, bbox: shf.Box, **kwargs: Any) -> Self:
        return cls(image=other.image[bbox], mask=other.mask[bbox], variance=other.variance[bbox], **kwargs)

    @pydantic.model_validator(mode="after")
    def _validate_bbox(self) -> Self:
        if self.image.bbox != self.mask.bbox:
            raise ValueError(
                f"Inconsistent bounding box between image ({self.image.bbox}) and mask ({self.mask.bbox})."
            )
        if self.image.bbox != self.variance.bbox:
            raise ValueError(
                f"Inconsistent bounding box between image ({self.image.bbox}) and "
                f"variance ({self.variance.bbox})."
            )
        return self

    @property
    def bbox(self) -> shf.Box:
        return self.image.bbox

    @classmethod
    def make_example(cls, bbox: shf.Box, rng: np.random.RandomState, **kwargs: Any) -> Self:
        mask_schema = shf.MaskSchema(
            [
                shf.MaskPlane("SAT", "saturated"),
                shf.MaskPlane("CR", "cosmic ray"),
                None,
                shf.MaskPlane("DETECTED", "above detection threshold"),
            ]
        )
        result = cls.from_bbox(bbox, mask_schema, dtype=np.int16, **kwargs)
        result.image.array = np.round(rng.randn(*result.bbox.size.shape) * 10)
        result.mask.array |= np.multiply.outer(result.image.array > 10.0, mask_schema.bitmask("DETECTED"))
        result.mask.array |= np.multiply.outer(result.image.array > 20.0, mask_schema.bitmask("SAT"))
        result.mask.array |= np.multiply.outer(
            rng.randn(*result.bbox.size.shape) > 2.0, mask_schema.bitmask("SAT")
        )
        return result


class Exposure(MaskedImage, ExposureInfo):
    @classmethod
    def make_example(cls, bbox: shf.Box, rng: np.random.RandomState, **kwargs: Any) -> Self:
        return super().make_example(
            bbox,
            rng,
            wcs=make_example_fits_wcs(bbox, rng),
            visit_info=VisitInfo.make_example(rng),
            photo_calib=PhotoCalib.make_example(bbox, rng),
            **kwargs,
        )


class Stamp(MaskedImage):
    object_id: Annotated[int, shf.ExportFitsHeaderKey("OBJID")]
    wcs: Annotated[WcsInterface, shf.Polymorphic(get_wcs_tag)]


class StampList(pydantic.BaseModel):
    stamps: list[Annotated[Stamp, shf.FitsOptions(subheader=True)]] = pydantic.Field(default_factory=list)
    visit_info: VisitInfo | None = None
    photo_calib: PhotoCalib | None = None

    @classmethod
    def make_example(cls, bbox: shf.Box, rng: np.random.RandomState, **kwargs: Any) -> Self:
        full_exposure = Exposure.make_example(bbox, rng, **kwargs)
        result = cls(visit_info=full_exposure.visit_info, photo_calib=full_exposure.photo_calib)
        box_x = rng.randint(bbox.x.start, bbox.x.stop, size=(5, 2))
        box_y = rng.randint(bbox.y.start, bbox.y.stop, size=(5, 2))
        for n in range(5):
            bbox = shf.Box(
                x=shf.Interval.hull(*box_x[n, :]),
                y=shf.Interval.hull(*box_y[n, :]),
            )
            result.stamps.append(
                Stamp.from_section(
                    full_exposure,
                    bbox,
                    wcs=AffineWcs.approximate(full_exposure.wcs, bbox.x.center, bbox.y.center),
                    object_id=n,
                )
            )
        return result
