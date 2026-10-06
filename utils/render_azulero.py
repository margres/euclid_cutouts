"""azulero colour rendering (Basset et al., CNES; pip install azulero>=2.0).

LRGB composite from all 4 Euclid bands: dead-pixel inpainting, per-band
sharpening, asinh stretch, band-to-colour mixing, hue/saturation mapping.

    from utils.render_azulero import render_azulero
    rgb = render_azulero(iyjh)                              # defaults
    rgb = render_azulero(iyjh, stretch=26.0, saturation=1.5)
"""

from __future__ import annotations

import numpy as np
from azulero.image import color as _az_color, mask as _az_mask
from azulero.tools.parsing import parse_map as _az_parse_map

# azulero 2.0 defaults. Per-band values are in IYJH order (VIS, NIR-Y, NIR-J, NIR-H).
ZERO_POINTS = (24.5, 29.8, 30.1, 30.0)   # MER MAGZERO values
SCALING     = (2.2, 1.3, 1.2, 1.0)
FWHM        = (1.6, 3.5, 3.4, 3.5)
# Per-channel curve knots in RGB order, azulero CLI syntax ("x: y, ..."; "" = identity).
CURVES      = ("", "", "0.5: 0.55")


def _parse_curves(curves):
    """Same knot handling as azulero's ``process`` CLI; returns BGR-ordered knot lists."""
    parsed = []
    for text in curves:
        knots = _az_parse_map(text)
        if knots:
            if knots[0][0] != 0 and knots[0][1] != 0:
                knots.insert(0, (0, 0))
            if knots[-1][0] != 1 and knots[-1][1] != 1:
                knots.append((1, 1))
        else:
            knots = [(0, 0), (1, 1)]
        parsed.append(knots)
    return parsed[::-1]   # RGB to BGR


def render_azulero(
    iyjh: np.ndarray,
    *,
    hue: float = -20,
    saturation: float = 1.2,
    stretch: float = 27.5,
    white: float = 22.5,
    offset: float = 28.5,
    sharpen: float = 0.5,
    nir_to_l: float = 0.1,
    i_to_b: float = 1.0,
    y_to_g: float = 0.5,
    j_to_r: float = 0.25,
    scaling=SCALING,
    zero_points=ZERO_POINTS,
    fwhm=FWHM,
    curves=CURVES,
) -> np.ndarray:
    """Render a (4, H, W) IYJH cutout (raw ADU/s) to (H, W, 3) uint8 RGB, north up.

    Parameters
    ----------
    hue, saturation : colour rotation (degrees) and saturation gain.
    stretch : stretching factor (AB mag); asinh softening is 10**((stretch - white)/2.5),
        so higher = stronger stretch, lower = closer to linear.
    white : white point (AB mag), or 0 for azulero's experimental auto-tuning.
    offset : opposite of the black point (AB mag).
    sharpen : per-band sharpening strength (0 = off).
    nir_to_l : fraction of NIR mixed into the luminance channel.
    i_to_b, y_to_g, j_to_r : VIS/Y/J contribution to B/G/R (0-1); the rest comes
        from the next-redder band (Y, J, H respectively).
    scaling, zero_points, fwhm : per-band (IYJH) flux scaling, zeropoints, PSF FWHM (px).
    curves : per-channel (R, G, B) spline knots applied last, e.g. "0.5: 0.55";
        "" = identity. Default boosts blue midtones; pass ("", "", "") to disable.
    """
    transform = _az_color.Transform(
        iyjh_zero_points=np.array(zero_points, dtype=np.float64),
        iyjh_scaling=np.array(scaling, dtype=np.float64),
        iyjh_fwhm=np.array(fwhm, dtype=np.float64),
        sharpen_strength=sharpen,
        nir_to_l=nir_to_l,
        i_to_b=i_to_b,
        y_to_g=y_to_g,
        j_to_r=j_to_r,
        hue=hue,
        saturation=saturation,
        stretch=stretch,
        bw=np.array([offset, white], dtype=np.float64),
        curves=_parse_curves(curves),
    )

    iyjh = iyjh.copy()
    dead = _az_mask.dead_pixels(iyjh)
    iyjh[0] = _az_mask.inpaint(iyjh[0], dead[0])
    nir_dead = dead[1] | dead[2] | dead[3]
    iyjh[1:] = _az_mask.inpaint(iyjh[1:], nir_dead, 0)
    iyjh = _az_color.sharpen(iyjh, transform.iyjh_fwhm / 2.355, transform.sharpen_strength)
    iyjh = _az_color.stretch_iyjh(iyjh, transform)
    lbgr = _az_color.iyjh_to_lbgr(iyjh, transform)
    bgr = _az_color.lbgr_to_bgr(lbgr, transform)
    bgr[dead[0]] = _az_mask.resaturate(bgr[dead[0]])
    for i, knots in enumerate(transform.curves):
        if knots != [(0, 0), (1, 1)]:
            bgr[..., i] = _az_color.adjust_curve(bgr[..., i], knots)
    rgb = np.flipud(np.clip(bgr[..., ::-1], 0.0, 1.0))
    return (rgb * 255).round().astype(np.uint8)
