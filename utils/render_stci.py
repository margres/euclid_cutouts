"""STCI colour rendering (SpaceTelescopeColorImage, Tian Li; pip install STCI).

PixInsight-like pipeline on VIS/Y/J (NIR-H unused): background
neutralisation, colour calibration, linked STF/histogram transform,
luminosity replacement, SCNR, saturation.

    from utils.render_stci import render_stci
    rgb = render_stci(iyjh)                                  # defaults
    rgb = render_stci(iyjh, saturation_amount=1.0, ht_dim_midtones=0.5)
"""

from __future__ import annotations

import numpy as np
from STCI import ComposeConfig, compose_pipeline, normalize_raw_channels_common


def render_stci(iyjh: np.ndarray, **config) -> np.ndarray:
    """Render a (4, H, W) IYJH cutout (raw ADU/s) to (H, W, 3) uint8 RGB, north up.

    Any keyword argument is passed to STCI's ``ComposeConfig`` (STCI's own
    defaults in brackets):

    shadows_clipping [-2.8]      STF shadows clipping, sigma (normalised MAD) from the median
    target_background [0.25]     target background of the linked STF auto-stretch
    bn_target_background [0.03]  background pedestal after BackgroundNeutralization
    ht_dim_midtones [0.625]      midtones of the post-stretch HT dimming pass (>0.5 darkens)
    scnr1_amount [0.5]           SCNR on the inverted image (removes magenta)
    scnr2_amount [0.8]           SCNR removing green
    saturation_amount [0.8]      final saturation boost, S' = S * (1 + amount)
    replace_luminance [True]     replace CIELab L* with the stretched blue channel (VIS)
    """
    red, green, blue, _ = normalize_raw_channels_common(
        iyjh[2].copy(), iyjh[1].copy(), iyjh[0].copy())   # J, Y, VIS
    outputs = compose_pipeline(red, green, blue, config=ComposeConfig(**config))
    final = outputs["13_final.tif"]
    return np.round(np.clip(np.flipud(final), 0.0, 1.0) * 255).astype(np.uint8)
