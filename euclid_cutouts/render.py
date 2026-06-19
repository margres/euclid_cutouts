"""
Euclid colour cutout rendering — importable library.

All rendering functions accept explicit parameters (no global CONFIG).
Can be used standalone from notebooks / scripts:

    from euclid_cutouts import render_cutout, render_fits_dir, load_fits_cutout

Or via the CLI wrapper ``scripts/make_colour_cutouts.py``.
"""

from __future__ import annotations

import glob
import logging
import multiprocessing as mp
import os
import sys
from typing import Literal

import numpy as np
from astropy.io import fits
from PIL import Image

log = logging.getLogger(__name__)

# ── Lazy optional imports ───────────────────────────────────────────────────
# azulero_render lives inside astronomaly-euclid and is not always on sys.path.
# bulk_euclid may or may not be installed. We import lazily so the package
# itself is pip-installable without either dependency present.

_azulero_render = None
_bulk_funcs = None
_stci_funcs = None


def _ensure_azulero(cutana_root: str | None = None):
    global _azulero_render
    if _azulero_render is not None:
        return
    if cutana_root and cutana_root not in sys.path:
        sys.path.insert(0, cutana_root)
    from cutana_datalabs import azulero_render as _mod
    _azulero_render = _mod


def _ensure_bulk(bulk_euclid_root: str | None = None):
    global _bulk_funcs
    if _bulk_funcs is not None:
        return
    if bulk_euclid_root and bulk_euclid_root not in sys.path:
        sys.path.insert(0, bulk_euclid_root)
    from bulk_euclid.utils.cutout_utils import (
        make_composite_cutout,
        make_triple_cutout,
        apply_MTF,
        replace_luminosity_channel,
    )
    from bulk_euclid.utils.morphology_utils_ou_mer import make_vis_only_cutout
    _bulk_funcs = {
        "make_composite_cutout": make_composite_cutout,
        "make_triple_cutout": make_triple_cutout,
        "apply_MTF": apply_MTF,
        "replace_luminosity_channel": replace_luminosity_channel,
        "make_vis_only_cutout": make_vis_only_cutout,
    }


def _ensure_stci():
    global _stci_funcs
    if _stci_funcs is not None:
        return
    from STCI import compose_pipeline, ComposeConfig, normalize_raw_channels_common
    _stci_funcs = {
        "compose_pipeline": compose_pipeline,
        "ComposeConfig": ComposeConfig,
        "normalize_raw_channels_common": normalize_raw_channels_common,
    }


# ── Band mapping ────────────────────────────────────────────────────────────

IYJH_BAND_INDEX = {"VIS": 0, "NIR_Y": 1, "NIR_J": 2, "NIR_H": 3}


# ── Single-cutout renderers ────────────────────────────────────────────────

def render_azulero(
    iyjh: np.ndarray,
    *,
    cutana_root: str | None = None,
) -> np.ndarray:
    """Render a (4, H, W) IYJH cutout to an (H, W, 3) uint8 RGB via azulero.

    Parameters
    ----------
    iyjh : (4, H, W) float32 array — VIS, NIR-Y, NIR-J, NIR-H.
    cutana_root : path to astronomaly-euclid (added to sys.path if needed).
    """
    _ensure_azulero(cutana_root)
    transform = _azulero_render.build_transform()
    return _azulero_render.render_rgb_uint8(iyjh, transform)


def render_stci(
    iyjh: np.ndarray,
) -> np.ndarray:
    """Render a (4, H, W) IYJH cutout to an (H, W, 3) uint8 RGB via STCI.

    STCI (SpaceTelescopeColorImage, Tian Li) applies a PixInsight-like
    pipeline: background neutralisation, colour calibration, linked
    STF/histogram transformation, L-replacement, SCNR, and saturation.

    Parameters
    ----------
    iyjh : (4, H, W) float32 array — VIS, NIR-Y, NIR-J, NIR-H.
        Only VIS (index 0), NIR-Y (1), and NIR-J (2) are used.
    """
    _ensure_stci()
    f = _stci_funcs
    vis = iyjh[0].copy()
    y_im = iyjh[1].copy()
    j_im = iyjh[2].copy()
    red, green, blue, _ = f["normalize_raw_channels_common"](j_im, y_im, vis)
    outputs = f["compose_pipeline"](red, green, blue, config=f["ComposeConfig"]())
    final = outputs["13_final.tif"]
    return np.round(np.clip(np.flipud(final), 0.0, 1.0) * 255).astype(np.uint8)


def render_bulk_variant(
    variant: str,
    vis: np.ndarray,
    y: np.ndarray | None = None,
    j: np.ndarray | None = None,
    *,
    bulk_euclid_root: str | None = None,
) -> np.ndarray | None:
    """Render one bulk_euclid colour variant from per-band 2-D arrays.

    Parameters
    ----------
    variant : one of ``gz_arcsinh_vis_y``, ``gz_arcsinh_vis_only``,
        ``gz_arcsinh_triple``, ``sw_mtf_vis_only``, ``sw_mtf_vis_y``,
        ``sw_mtf_vis_y_j``.
    vis, y, j : (H, W) float32 arrays.  ``y`` and ``j`` may be None when
        the variant doesn't need them.
    bulk_euclid_root : path to bulk-euclid-cutouts clone.

    Returns
    -------
    (H, W, 3) uint8 RGB with north up (flipud applied), or None if the
    variant name is unrecognised.
    """
    _ensure_bulk(bulk_euclid_root)
    f = _bulk_funcs
    rgb = None

    if variant == "gz_arcsinh_vis_y":
        rgb = f["make_composite_cutout"](vis.copy(), y.copy(), vis_q=100, nisp_q=0.2)

    elif variant == "gz_arcsinh_vis_only":
        grey = f["make_vis_only_cutout"](vis.copy(), q=100)
        rgb = np.stack([grey, grey, grey], axis=2)

    elif variant == "gz_arcsinh_triple":
        rgb = f["make_triple_cutout"](vis.copy(), y.copy(), j.copy(),
                                      short_q=100, mid_q=0.2, long_q=0.1)

    elif variant == "sw_mtf_vis_only":
        vis_mtf = f["apply_MTF"](vis.copy())
        rgb = np.stack([vis_mtf, vis_mtf, vis_mtf], axis=2)

    elif variant == "sw_mtf_vis_y":
        vis_mtf = f["apply_MTF"](vis.copy())
        y_mtf = f["apply_MTF"](y.copy())
        mean_mtf = np.mean([vis_mtf, y_mtf], axis=0).astype(np.uint8)
        rgb = np.stack([y_mtf, mean_mtf, vis_mtf], axis=2)
        rgb = f["replace_luminosity_channel"](rgb, rgb_channel_for_luminosity=2,
                                               desaturate_speckles=False)

    elif variant == "sw_mtf_vis_y_j":
        vis_mtf = f["apply_MTF"](vis.copy())
        y_mtf = f["apply_MTF"](y.copy())
        j_mtf = f["apply_MTF"](j.copy())
        rgb = np.stack([j_mtf, y_mtf, vis_mtf], axis=2)
        rgb = f["replace_luminosity_channel"](rgb, rgb_channel_for_luminosity=2,
                                               desaturate_speckles=False)

    else:
        log.warning("Unknown bulk_euclid variant: %s", variant)
        return None

    return np.flipud(rgb)


def render_cutout(
    iyjh: np.ndarray,
    *,
    renderers: list[str] | None = None,
    bulk_variants: list[str] | None = None,
    cutana_root: str | None = None,
    bulk_euclid_root: str | None = None,
) -> dict[str, np.ndarray]:
    """Render a single (4, H, W) IYJH cutout through one or more colour stretches.

    Parameters
    ----------
    iyjh : (4, H, W) float32 — VIS, NIR-Y, NIR-J, NIR-H.
    renderers : which renderers to apply.  Accepted values: ``"azulero"``,
        ``"stci"``, ``"bulk_euclid"``.  Defaults to ``["azulero"]``.
    bulk_variants : which bulk_euclid variants to produce (ignored unless
        ``"bulk_euclid"`` is in *renderers*).  Defaults to
        ``["gz_arcsinh_vis_y"]``.
    cutana_root : path to astronomaly-euclid.
    bulk_euclid_root : path to bulk-euclid-cutouts.

    Returns
    -------
    dict mapping renderer/variant name to (H, W, 3) uint8 RGB.
    """
    if renderers is None:
        renderers = ["azulero"]
    if bulk_variants is None:
        bulk_variants = ["gz_arcsinh_vis_y"]

    results: dict[str, np.ndarray] = {}

    if "azulero" in renderers:
        results["azulero"] = render_azulero(iyjh, cutana_root=cutana_root)

    if "stci" in renderers:
        results["stci"] = render_stci(iyjh)

    if "bulk_euclid" in renderers:
        vis = iyjh[0]
        y_im = iyjh[1]
        j_im = iyjh[2]
        for variant in bulk_variants:
            rgb = render_bulk_variant(variant, vis, y_im, j_im,
                                     bulk_euclid_root=bulk_euclid_root)
            if rgb is not None:
                results[variant] = rgb

    return results


# ── FITS-cutout I/O ─────────────────────────────────────────────────────────

def load_fits_cutout(
    path: str,
    band_order: list[str] | None = None,
) -> np.ndarray | None:
    """Load a multi-extension FITS cutout into a (4, H, W) IYJH float32 array.

    Parameters
    ----------
    path : path to the FITS file.
    band_order : list mapping extension index to band name, e.g.
        ``["VIS", "NIR_Y", "NIR_J", "NIR_H"]``.  Defaults to that 4-band order.

    Returns None if no usable image extensions are found.
    """
    if band_order is None:
        band_order = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]

    with fits.open(path) as hdul:
        img_hdus = [h for h in hdul if h.data is not None and h.data.ndim == 2]
        if not img_hdus:
            return None

        h, w = img_hdus[0].data.shape
        iyjh = np.zeros((4, h, w), dtype=np.float32)

        for i, hdu in enumerate(img_hdus):
            if i >= len(band_order):
                break
            idx = IYJH_BAND_INDEX.get(band_order[i])
            if idx is not None:
                iyjh[idx] = hdu.data.astype(np.float32)

    return iyjh


# ── Batch: directory of FITS cutouts ────────────────────────────────────────

def _init_worker():
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


# Module-level dict used to pass config to pool workers (set by render_fits_dir).
_worker_cfg: dict = {}


def _render_one_fits(fits_path: str) -> tuple[str, dict[str, int]]:
    cfg = _worker_cfg
    stem = os.path.splitext(os.path.basename(fits_path))[0]
    counts: dict[str, int] = {}

    iyjh = load_fits_cutout(fits_path, band_order=cfg["band_order"])
    if iyjh is None:
        log.warning("No image data in %s — skipping", fits_path)
        return stem, counts

    fmt = cfg["fmt"]
    save_kw = {"quality": cfg["jpeg_quality"]} if fmt == "jpg" else {}

    if cfg["enable_azulero"]:
        out_path = os.path.join(cfg["azulero_out"], f"{stem}.{fmt}")
        if not os.path.exists(out_path):
            try:
                rgb = render_azulero(iyjh, cutana_root=cfg["cutana_root"])
                Image.fromarray(rgb).save(out_path, **save_kw)
                counts["azulero"] = 1
            except Exception:
                log.exception("  azulero failed for %s", stem)

    if cfg["enable_stci"]:
        out_path = os.path.join(cfg["stci_out"], f"{stem}.{fmt}")
        if not os.path.exists(out_path):
            try:
                rgb = render_stci(iyjh)
                Image.fromarray(rgb).save(out_path, **save_kw)
                counts["stci"] = 1
            except Exception:
                log.exception("  stci failed for %s", stem)

    if cfg["enable_bulk"]:
        try:
            import cv2
            cv2.setNumThreads(1)
        except ImportError:
            pass
        vis, y_im, j_im = iyjh[0], iyjh[1], iyjh[2]
        for variant, out_dir in cfg["bulk_out_dirs"].items():
            out_path = os.path.join(out_dir, f"{stem}.{fmt}")
            if os.path.exists(out_path):
                continue
            try:
                rgb = render_bulk_variant(variant, vis, y_im, j_im,
                                         bulk_euclid_root=cfg["bulk_euclid_root"])
                if rgb is None:
                    continue
                Image.fromarray(rgb).save(out_path, **save_kw)
                counts[variant] = counts.get(variant, 0) + 1
            except Exception:
                log.exception("  bulk_euclid %s failed for %s", variant, stem)

    return stem, counts


def render_fits_dir(
    input_dir: str,
    output_dir: str = "cutouts",
    *,
    enable_azulero: bool = True,
    enable_stci: bool = True,
    enable_bulk_euclid: bool = True,
    bulk_variants: list[str] | None = None,
    band_order: list[str] | None = None,
    fmt: Literal["jpg", "png"] = "jpg",
    jpeg_quality: int = 95,
    n_workers: int = 1,
    progress_bar: bool = False,
    cutana_root: str | None = None,
    bulk_euclid_root: str | None = None,
) -> dict[str, int]:
    """Colour-render a directory of FITS cutout files.

    Parameters
    ----------
    input_dir : directory containing ``*.fits`` cutout files.
    output_dir : root output directory (subdirs are created per renderer).
    enable_azulero : produce azulero colour images.
    enable_stci : produce STCI colour images.
    enable_bulk_euclid : produce bulk_euclid colour images.
    bulk_variants : which bulk_euclid variants.  Defaults to
        ``["gz_arcsinh_vis_y"]``.
    band_order : maps FITS extension index to band.  Defaults to
        ``["VIS", "NIR_Y", "NIR_J", "NIR_H"]``.
    fmt : output image format (``"jpg"`` or ``"png"``).
    jpeg_quality : JPEG quality (1–100).
    n_workers : number of parallel workers.
    progress_bar : show a tqdm progress bar.
    cutana_root : path to astronomaly-euclid (for azulero).
    bulk_euclid_root : path to bulk-euclid-cutouts.

    Returns
    -------
    dict mapping renderer/variant name to total count produced.
    """
    if bulk_variants is None:
        bulk_variants = ["gz_arcsinh_vis_y"]
    if band_order is None:
        band_order = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]

    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    if not fits_files:
        log.error("No .fits files found in %s", input_dir)
        return {}
    log.info("FITS-input mode: %d files in %s", len(fits_files), input_dir)

    azulero_out = os.path.join(output_dir, "azulero")
    if enable_azulero:
        os.makedirs(azulero_out, exist_ok=True)

    stci_out = os.path.join(output_dir, "stci")
    if enable_stci:
        os.makedirs(stci_out, exist_ok=True)

    bulk_out_dirs: dict[str, str] = {}
    if enable_bulk_euclid:
        for variant in bulk_variants:
            d = os.path.join(output_dir, variant)
            os.makedirs(d, exist_ok=True)
            bulk_out_dirs[variant] = d

    active = []
    if enable_azulero:
        active.append("azulero")
    if enable_stci:
        active.append("stci")
    if bulk_out_dirs:
        active.extend(bulk_variants)
    log.info("Active renderers: %s", ", ".join(active))
    log.info("Band mapping: %s", " → ".join(
        f"CHANNEL_{i+1}={b}" for i, b in enumerate(band_order)))

    global _worker_cfg
    _worker_cfg = {
        "band_order": band_order,
        "enable_azulero": enable_azulero,
        "enable_stci": enable_stci,
        "enable_bulk": enable_bulk_euclid and bool(bulk_out_dirs),
        "azulero_out": azulero_out,
        "stci_out": stci_out,
        "bulk_out_dirs": bulk_out_dirs,
        "fmt": fmt,
        "jpeg_quality": jpeg_quality,
        "cutana_root": cutana_root,
        "bulk_euclid_root": bulk_euclid_root,
    }

    totals: dict[str, int] = {}
    done = 0
    n_total = len(fits_files)

    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(_render_one_fits, fits_files)
        if progress_bar:
            from tqdm import tqdm
            results_iter = tqdm(results_iter, total=n_total, unit="file",
                                desc="Rendering FITS cutouts")
        for stem, counts in results_iter:
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            done += 1
            if progress_bar:
                results_iter.set_postfix_str(
                    "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))
            elif done % 50 == 0 or done == n_total:
                log.info("[%d/%d files] %s", done, n_total,
                         "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))

    lines = [f"Done. {done} files processed"]
    for k in sorted(totals):
        d = azulero_out if k == "azulero" else bulk_out_dirs.get(k, output_dir)
        lines.append(f"  {k}={totals[k]} → {d}/")
    log.info("\n".join(lines))

    return totals
