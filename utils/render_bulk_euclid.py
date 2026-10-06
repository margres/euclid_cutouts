"""bulk_euclid colour rendering (Walmsley, M., bulk-euclid-cutouts).

Arcsinh (``gz_*``, Galaxy Zoo style) and midtone-transfer-function
(``sw_mtf_*``, Space Warps style) stretches, VIS-only up to VIS+Y+J.
bulk_euclid isn't on PyPI: clone https://github.com/mwalmsley/bulk-euclid-cutouts
and pass its path as ``bulk_euclid_root`` (or pip install -e it).

    from utils.render_bulk_euclid import render_bulk_euclid, VARIANTS
    rgb = render_bulk_euclid(iyjh, "sw_mtf_vis_y_j", bulk_euclid_root="bulk-euclid-cutouts")
    rgb = render_bulk_euclid(iyjh, "sw_mtf_vis_y_j", mtf_mean=0.3, bulk_euclid_root=...)
"""

from __future__ import annotations

import sys

import numpy as np

VARIANTS = [
    "gz_arcsinh_vis_y",     # VIS+Y composite (arcsinh)
    "gz_arcsinh_vis_only",  # VIS-only (arcsinh)
    "gz_arcsinh_triple",    # VIS+Y+J three-band (arcsinh)
    "sw_mtf_vis_only",      # VIS-only (midtone transfer)
    "sw_mtf_vis_y",         # VIS+Y (midtone transfer + LAB luminosity)
    "sw_mtf_vis_y_j",       # VIS+Y+J (midtone transfer + LAB luminosity)
]


def _import_bulk(bulk_euclid_root):
    if bulk_euclid_root and bulk_euclid_root not in sys.path:
        sys.path.insert(0, bulk_euclid_root)
    from bulk_euclid.utils import cutout_utils, morphology_utils_ou_mer
    return cutout_utils, morphology_utils_ou_mer


def render_bulk_euclid(
    iyjh: np.ndarray,
    variant: str,
    *,
    bulk_euclid_root: str | None = None,
    vis_q: float = 100,
    y_q: float = 0.2,
    j_q: float = 0.1,
    clip: float = 99.85,
    mtf_mean: float = 0.2,
    desaturate_speckles: bool = False,
) -> np.ndarray:
    """Render a (4, H, W) IYJH cutout (raw ADU/s) to (H, W, 3) uint8 RGB, north up.

    Parameters
    ----------
    variant : one of ``VARIANTS``.
    bulk_euclid_root : path to a bulk-euclid-cutouts clone.
    vis_q, y_q, j_q : per-band stretch ``arcsinh(flux * q)`` (``gz_*`` only); higher =
        stronger stretch of faint flux. ``gz_arcsinh_vis_y`` uses vis_q/y_q.
    clip : upper percentile clip applied after the arcsinh stretch (``gz_*`` only).
    mtf_mean : target mean (0-1) of the central 100x100 px after min-max
        normalisation; the MTF midtone is solved to hit it (``sw_mtf_*`` only).
    desaturate_speckles : after the LAB luminosity swap, cap HSV saturation at
        150/255 and halve the brightness of pixels above it (every pixel, not a
        local test) (``sw_mtf_vis_y`` / ``sw_mtf_vis_y_j`` only).
    """
    cu, mu = _import_bulk(bulk_euclid_root)
    vis, y, j = iyjh[0].copy(), iyjh[1].copy(), iyjh[2].copy()

    if variant == "gz_arcsinh_vis_y":
        rgb = cu.make_composite_cutout(vis, y, vis_q=vis_q, vis_clip=clip,
                                       nisp_q=y_q, nisp_clip=clip)
    elif variant == "gz_arcsinh_vis_only":
        grey = mu.make_vis_only_cutout(vis, q=vis_q, clip=clip)
        rgb = np.stack([grey] * 3, axis=2)
    elif variant == "gz_arcsinh_triple":
        rgb = cu.make_triple_cutout(vis, y, j, short_q=vis_q, mid_q=y_q, long_q=j_q,
                                    short_clip=clip, mid_clip=clip, long_clip=clip)
    elif variant == "sw_mtf_vis_only":
        v = cu.apply_MTF(vis, desired_mean_normalized=mtf_mean)
        rgb = np.stack([v] * 3, axis=2)
    elif variant == "sw_mtf_vis_y":
        v = cu.apply_MTF(vis, desired_mean_normalized=mtf_mean)
        yy = cu.apply_MTF(y, desired_mean_normalized=mtf_mean)
        mean = np.mean([v, yy], axis=0).astype(np.uint8)
        rgb = cu.replace_luminosity_channel(np.stack([yy, mean, v], axis=2),
                                            rgb_channel_for_luminosity=2,
                                            desaturate_speckles=desaturate_speckles)
    elif variant == "sw_mtf_vis_y_j":
        v = cu.apply_MTF(vis, desired_mean_normalized=mtf_mean)
        yy = cu.apply_MTF(y, desired_mean_normalized=mtf_mean)
        jj = cu.apply_MTF(j, desired_mean_normalized=mtf_mean)
        rgb = cu.replace_luminosity_channel(np.stack([jj, yy, v], axis=2),
                                            rgb_channel_for_luminosity=2,
                                            desaturate_speckles=desaturate_speckles)
    else:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")

    return np.flipud(rgb)
