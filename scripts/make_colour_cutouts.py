"""
Produce colour cutouts for Euclid sources using four renderers:
azulero, eummy, bulk_euclid GZ arcsinh, and bulk_euclid SW MTF.

Works with Q1 and DR1. Input is a CSV with at minimum ra and dec columns.
See CONFIG block for all options.

Run:
    python make_colour_cutouts.py
"""

import glob
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

_BULK_EUCLID_ROOT = "/media/user/bulk-euclid-cutouts"
if _BULK_EUCLID_ROOT not in sys.path:
    sys.path.insert(0, _BULK_EUCLID_ROOT)

sys.path.insert(0, os.path.dirname(__file__))

from fits_path_utils import find_fits_paths_any_release, find_fits_paths, find_iyjh_paths  # noqa: E402
from euclid_cutouts.render import build_azulero_transform, _render_azulero_rgb  # noqa: E402
from bulk_euclid.utils.cutout_utils import (  # noqa: E402
    make_composite_cutout,
    make_triple_cutout,
    apply_MTF,
    replace_luminosity_channel,
)
from bulk_euclid.utils.morphology_utils_ou_mer import (  # noqa: E402
    make_vis_only_cutout,
)
from STCI import compose_pipeline, ComposeConfig, normalize_raw_channels_common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Input CSV.
# Required columns : RA, Dec (auto-detected; see _RA_ALIASES / _DEC_ALIASES)
# Optional columns :
#   id              — used as filename stem when NAMING="id" (defaults to row index)
#   object_id       — MER catalog object ID; required when NAMING="q1_slde"
#   tile_index      — Euclid tile ID (auto-looked-up from TILE_CENTRES_CSV if absent)
#   size_pixel      — cutout size in VIS pixels, per source
#   size_arcsec     — cutout size in arcsec, per source (size_pixel takes precedence)
SOURCES_CSV           = "sources.csv"

# Manual RA/Dec column override. Set to None for auto-detection (recommended),
# or to your column names if the CSV uses non-standard names.
RA_COL                = None       # e.g. "my_ra_column"
DEC_COL               = None       # e.g. "my_dec_column"

# HEALPix tile map — used to resolve tile_index from ra/dec when absent from the CSV.
# tile_centres.csv is used only to enrich tile_index with release_dir.
HEALPIX_MAP      = "/media/home/my_workspace/euclid_cutouts/data/tile_index_map.v1.2.fits.gz"
TILE_CENTRES_CSV = "/media/home/my_workspace/euclid_cutouts/tile_centres.csv"

# Output root directory. All outputs share this root, in subfolders named
# after the format/stretch:
#   {OUTPUT_DIR}/fits/           — FITS cutouts (Cutana)
#   {OUTPUT_DIR}/azulero/        — colour images (azulero stretch)
#   {OUTPUT_DIR}/eummy/          — colour images (eummy stretch)
#   {OUTPUT_DIR}/{variant}/      — one subfolder per BULK_EUCLID_OUTPUTS entry
# All are flat (no per-tile subdirs), mirroring Cutana's default structure.
OUTPUT_DIR            = "cutouts"

# Output image format for colour cutouts.
#   "jpg"  — lossy JPEG  (~8 KB/cutout, ~8 GB per 1M cutouts)
#   "png"  — lossless PNG (~60 KB/cutout, ~63 GB per 1M cutouts)
# Enable/disable each renderer independently.
ENABLE_AZULERO        = True
ENABLE_EUMMY          = True
ENABLE_STCI           = True
ENABLE_BULK_EUCLID    = True

AZULERO_FORMAT        = "jpg"
EUMMY_FORMAT          = "png"
STCI_FORMAT           = "jpg"
BULK_EUCLID_FORMAT    = "jpg"
JPEG_QUALITY          = 95

# bulk_euclid colour outputs to produce.
# GZ (Galaxy Zoo) arcsinh variants:
#   "gz_arcsinh_vis_y"     — VIS+Y composite (default Galaxy Zoo rendering)
#   "gz_arcsinh_vis_only"  — VIS-only arcsinh
#   "gz_arcsinh_triple"    — VIS+Y+J three-band composite
# SW (Space Warps) MTF variants:
#   "sw_mtf_vis_only"      — VIS-only midtone transfer
#   "sw_mtf_vis_y"         — VIS+Y MTF with LAB luminosity replacement
#   "sw_mtf_vis_y_j"       — VIS+Y+J MTF (best for strong lensing)
BULK_EUCLID_OUTPUTS   = [
    "gz_arcsinh_vis_y",
]

# File naming convention for output cutouts.
#   "id"            — use the id column (or row index if absent)
#   "q1_slde"       — {tile_index}_{object_id}  (requires object_id column)
#   "cutana_default"— {id}_{ra:.6f}_{dec:.6f}   (mirrors Cutana's built-in template)
NAMING                = "id"

# Default cutout size when not specified per source
DEFAULT_CUTOUT_PIXELS = 101
VIS_ARCSEC_PER_PX     = 0.1   # VIS plate scale (arcsec per pixel)

N_WORKERS             = 1
PROGRESS_BAR          = False  # set True to show a tqdm progress bar (requires `pip install tqdm`)

# ── FITS-input mode ────────────────────────────────────────────────────────
# When INPUT_FITS_DIR is set to a directory path, the pipeline reads pre-made
# FITS cutout files from that directory (instead of loading tile FITS and
# extracting cutouts). Each file should be a multi-extension FITS with one
# ImageHDU per band. The band order is specified by FITS_BAND_ORDER.
# Eummy is not supported in this mode (it needs tile-level FITS).
INPUT_FITS_DIR        = None   # e.g. "/media/user/euclid_cutouts/results/200k/cutouts"

# Band order in the input FITS extensions. Maps extension index (CHANNEL_1,
# CHANNEL_2, ...) to band identity. Use "VIS", "NIR_Y", "NIR_J", "NIR_H".
# Missing bands (fewer extensions than 4) are zero-filled.
FITS_BAND_ORDER       = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]

# Fallback release dirs — searched in order when a tile is not in TILE_CENTRES_CSV.
# Q1 (R1 only):
RELEASE_DIRS          = ["/media/home/data/euclid_q1/Q1_R1"]
# DR1 (comment Q1 line above and uncomment below):
# RELEASE_DIRS        = ["/media/home/data/euclid_idr1/DR1/R2",
#                        "/media/home/data/euclid_idr1/DR1/R1"]

BANDS_3               = ["VIS", "NIR_Y", "NIR_J"]  # NIR_H derived by find_iyjh_paths
BAND_TO_INST          = {"VIS": "VIS", "NIR_Y": "NISP", "NIR_J": "NISP"}
# ── END CONFIG ────────────────────────────────────────────────────────────────


def _make_stem(src: dict) -> str:
    """Return the output filename stem for a source according to NAMING."""
    if NAMING == "q1_slde":
        return f"{src['tile_index']}_{src['object_id']}"
    if NAMING == "cutana_default":
        return f"{src['id']}_{src['ra']:.6f}_{src['dec']:.6f}"
    return str(src["id"])  # "id" (default)


def _lookup_tile_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add tile_index (and release_dir if TILE_CENTRES_CSV is set) via HEALPix map.

    Uses the official Euclid tiling HEALPix map (order 13, nested). Sources
    outside any tile get tile_index=None and are dropped by load_sources.
    """
    import healpy

    healpix_array = healpy.read_map(HEALPIX_MAP, nest=True)

    hp_theta = np.pi / 2.0 - np.deg2rad(df["dec"].values)
    hp_phi   = np.deg2rad(df["ra"].values)
    nside    = healpy.order2nside(13)   # MOC order 13 — fixed by Euclid tiling v1.2
    pix      = healpy.ang2pix(nside, hp_theta, hp_phi, nest=True)
    tidx     = healpix_array[pix]

    if tidx.dtype == np.dtype(">i4"):   # big-endian int32 on some platforms
        tidx = tidx.byteswap().newbyteorder()

    df = df.copy()
    df["tile_index"] = np.where(tidx == 0, None, tidx.astype(object))

    if TILE_CENTRES_CSV:
        try:
            tc = pd.read_csv(TILE_CENTRES_CSV, usecols=["tile_index", "release_dir"])
            tc = tc.drop_duplicates("tile_index")
            df = df.merge(tc, on="tile_index", how="left")
        except Exception:
            pass

    return df


_RA_ALIASES  = ["ra", "right_ascension", "target_ra"]
_DEC_ALIASES = ["dec", "declination", "target_dec"]


def _normalise_radec(df: pd.DataFrame) -> pd.DataFrame:
    """Rename RA/Dec columns to 'ra' and 'dec'.

    Uses RA_COL/DEC_COL if set, otherwise auto-detects from _RA_ALIASES/_DEC_ALIASES.
    """
    if RA_COL and DEC_COL:
        ra_col = next((c for c in df.columns if c.lower().strip() == RA_COL.lower().strip()), None)
        dec_col = next((c for c in df.columns if c.lower().strip() == DEC_COL.lower().strip()), None)
        if ra_col is None or dec_col is None:
            raise ValueError(f"RA_COL={RA_COL!r} or DEC_COL={DEC_COL!r} not found in {list(df.columns)}")
    else:
        cols_lower = {c.lower().strip(): c for c in df.columns}
        ra_col = next((cols_lower[a] for a in _RA_ALIASES if a in cols_lower), None)
        dec_col = next((cols_lower[a] for a in _DEC_ALIASES if a in cols_lower), None)
        if ra_col is None or dec_col is None:
            raise ValueError(
                f"CSV must have RA and Dec columns (accepted: {_RA_ALIASES} / {_DEC_ALIASES}), "
                f"found: {list(df.columns)}"
            )
    return df.rename(columns={ra_col: "ra", dec_col: "dec"})


def load_sources(csv_path: str) -> pd.DataFrame:
    """Load and normalise sources from csv_path.

    Returns DataFrame with columns: id, tile_index, ra, dec, size_pixel, release_dir.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    df = _normalise_radec(df)

    # id — filename stem; if absent, defaults to row index
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    else:
        df["id"] = df.index.astype(str)

    # size_pixel — per source or global default
    if "size_pixel" not in df.columns:
        if "size_arcsec" in df.columns:
            df["size_pixel"] = (df["size_arcsec"] / VIS_ARCSEC_PER_PX).round().astype(int)
        else:
            df["size_pixel"] = DEFAULT_CUTOUT_PIXELS
    df["size_pixel"] = df["size_pixel"].astype(int)

    # tile_index — from CSV or spatial lookup
    if "tile_index" not in df.columns:
        logging.info("tile_index not in CSV — looking up via HEALPix map %s", HEALPIX_MAP)
        df = _lookup_tile_indices(df)
        n_miss = df["tile_index"].isna().sum()
        if n_miss:
            logging.warning("%d sources had no matching tile and will be dropped", n_miss)
        df = df.dropna(subset=["tile_index"]).reset_index(drop=True)
    else:
        df["tile_index"] = df["tile_index"].astype(int)
        # Enrich with release_dir from tile_centres (tile_index is unique across releases)
        if "release_dir" not in df.columns and TILE_CENTRES_CSV:
            try:
                tc = pd.read_csv(TILE_CENTRES_CSV, usecols=["tile_index", "release_dir"])
                df = df.merge(tc, on="tile_index", how="left")
            except Exception:
                pass

    df["tile_index"] = df["tile_index"].astype(int)
    if "release_dir" not in df.columns:
        df["release_dir"] = None

    return df[["id", "tile_index", "ra", "dec", "size_pixel", "release_dir"]]


def resolve_iyjh_paths(tile_id: int, ra: float, dec: float,
                        release_dir: str | None = None) -> list[str] | None:
    """Resolve 4 IYJH FITS paths for tile_id.

    Tries release_dir first (from tile_centres), then falls back to RELEASE_DIRS.
    Returns None if any band is missing.
    """
    search_dirs = []
    if release_dir:
        search_dirs.append(release_dir)
    # Add any RELEASE_DIRS not already included
    seen = set(search_dirs)
    for d in RELEASE_DIRS:
        if d not in seen:
            search_dirs.append(d)

    paths_3 = find_fits_paths_any_release(
        tile_id, BANDS_3, search_dirs, BAND_TO_INST, ra=ra, dec=dec
    )
    if paths_3 is None:
        return None
    return find_iyjh_paths(paths_3)


def _extract_cutout(iyjh: np.ndarray, wcs: WCS,
                    ra: float, dec: float, size: int) -> np.ndarray:
    """Slice a size×size cutout from a pre-loaded (4,H,W) array, zero-padding edges."""
    ny, nx = iyjh.shape[1], iyjh.shape[2]
    x, y = wcs.world_to_pixel_values(ra, dec)
    half = size // 2
    x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
    x1, y1 = x0 + size, y0 + size
    x0c, x1c = max(0, x0), min(nx, x1)
    y0c, y1c = max(0, y0), min(ny, y1)
    out = np.zeros((4, size, size), dtype=np.float32)
    if x1c > x0c and y1c > y0c:
        out[:, y0c - y0:y0c - y0 + (y1c - y0c),
               x0c - x0:x0c - x0 + (x1c - x0c)] = iyjh[:, y0c:y1c, x0c:x1c]
    return out


_EUMMY_RE = re.compile(r"^.+_(-?\d+\.\d+)([+-]\d+\.\d+)\.png$")


def _find_eummy_exe() -> str:
    exe = shutil.which("eummy")
    if exe:
        return exe
    candidate = os.path.join(os.path.dirname(sys.executable), "eummy")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "Could not find 'eummy'. Install it with 'pip install eummy'."
    )


def _setup_tile_workspace(tile_dir: str, iyjh_paths: list[str]) -> None:
    os.makedirs(tile_dir, exist_ok=True)
    for path in iyjh_paths:
        link = os.path.join(tile_dir, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(path, link)


def _write_tile_catalog(sources: list[dict], catalog_path: str) -> None:
    ra_arr  = np.array([s["ra"]  for s in sources], dtype=np.float64)
    dec_arr = np.array([s["dec"] for s in sources], dtype=np.float64)
    fits.BinTableHDU.from_columns([
        fits.Column(name="RA",  format="D", array=ra_arr),
        fits.Column(name="DEC", format="D", array=dec_arr),
    ]).writeto(catalog_path, overwrite=True)


def rename_eummy_cutouts(tile_dir: str, sources: list[dict],
                         eummy_out_dir: str,
                         match_tol_deg: float = 1.5 / 3600) -> int:
    """Match eummy PNGs to sources by RA/Dec, convert if needed, and save."""
    fmt = EUMMY_FORMAT.lower()
    ra_arr  = np.array([s["ra"]  for s in sources])
    dec_arr = np.array([s["dec"] for s in sources])

    os.makedirs(eummy_out_dir, exist_ok=True)
    n_renamed = 0
    for png_path in glob.glob(os.path.join(tile_dir, "TILE*_*.png")):
        fname = os.path.basename(png_path)
        match = _EUMMY_RE.match(fname)
        if not match:
            continue
        ra, dec = float(match.group(1)), float(match.group(2))

        dra  = (ra_arr - ra) * np.cos(np.radians(dec))
        ddec = dec_arr - dec
        dist = np.hypot(dra, ddec)
        i = int(np.argmin(dist))
        if dist[i] > match_tol_deg:
            logging.warning("  No source within match tolerance for %s (closest %.2f\")",
                            fname, dist[i] * 3600)
            continue

        dest = os.path.join(eummy_out_dir, f"{_make_stem(sources[i])}.{fmt}")
        if os.path.exists(dest):
            continue
        if fmt == "png":
            os.replace(png_path, dest)
        else:
            Image.open(png_path).convert("RGB").save(dest, quality=JPEG_QUALITY)
            os.remove(png_path)
        n_renamed += 1
    return n_renamed


def render_azulero_tile(iyjh: np.ndarray, wcs: WCS,
                        sources: list[dict], out_dir: str) -> int:
    """Render azulero images for all sources in one tile. Returns number written."""
    fmt = AZULERO_FORMAT.lower()
    transform = build_azulero_transform()
    n_ok = n_fail = 0
    for src in sources:
        out_path = os.path.join(out_dir, f"{_make_stem(src)}.{fmt}")
        if os.path.exists(out_path):
            continue
        try:
            size   = int(src.get("size_pixel", DEFAULT_CUTOUT_PIXELS))
            cutout = _extract_cutout(iyjh, wcs, src["ra"], src["dec"], size)
            rgb_f  = _render_azulero_rgb(cutout, transform)
            rgb    = (np.clip(rgb_f, 0.0, 1.0) * 255).round().astype(np.uint8)
            save_kw = {"quality": JPEG_QUALITY} if fmt == "jpg" else {}
            Image.fromarray(rgb).save(out_path, **save_kw)
            n_ok += 1
        except Exception:
            logging.exception("  azulero render failed for %s", src["id"])
            n_fail += 1

    if n_fail:
        logging.warning("  %d renders failed in %s", n_fail, out_dir)
    return n_ok


def render_stci_tile(iyjh: np.ndarray, wcs: WCS,
                     sources: list[dict], out_dir: str) -> int:
    """Render STCI images for all sources in one tile. Returns number written."""
    fmt = STCI_FORMAT.lower()
    config = ComposeConfig()
    n_ok = n_fail = 0
    for src in sources:
        out_path = os.path.join(out_dir, f"{_make_stem(src)}.{fmt}")
        if os.path.exists(out_path):
            continue
        try:
            size = int(src.get("size_pixel", DEFAULT_CUTOUT_PIXELS))
            cutout = _extract_cutout(iyjh, wcs, src["ra"], src["dec"], size)
            vis, y_im, j_im = cutout[0], cutout[1], cutout[2]
            red, green, blue, _ = normalize_raw_channels_common(j_im, y_im, vis)
            outputs = compose_pipeline(red, green, blue, config=config)
            final = outputs["13_final.tif"]
            final_uint8 = np.round(np.clip(np.flipud(final), 0.0, 1.0) * 255).astype(np.uint8)
            save_kw = {"quality": JPEG_QUALITY} if fmt == "jpg" else {}
            Image.fromarray(final_uint8).save(out_path, **save_kw)
            n_ok += 1
        except Exception:
            logging.exception("  STCI render failed for %s", src["id"])
            n_fail += 1

    if n_fail:
        logging.warning("  %d STCI renders failed in %s", n_fail, out_dir)
    return n_ok


def render_bulk_euclid_tile(iyjh: np.ndarray, wcs: WCS,
                           sources: list[dict],
                           out_dirs: dict[str, str]) -> dict[str, int]:
    """Render bulk_euclid GZ/SW colour images for all sources in one tile.

    out_dirs maps variant name (e.g. "gz_arcsinh_vis_y") to its output directory.
    Returns {variant: n_written}.
    """
    import cv2  # noqa: F811 — needed by replace_luminosity_channel
    cv2.setNumThreads(1)

    counts: dict[str, int] = {v: 0 for v in out_dirs}
    if not out_dirs:
        return counts

    needs_y = any(v for v in out_dirs if "vis_y" in v)
    needs_j = any(v for v in out_dirs if "vis_y_j" in v or "triple" in v)

    fmt = BULK_EUCLID_FORMAT.lower()
    save_kw = {"quality": JPEG_QUALITY} if fmt == "jpg" else {}

    for src in sources:
        stem = _make_stem(src)
        size = int(src.get("size_pixel", DEFAULT_CUTOUT_PIXELS))
        cutout = _extract_cutout(iyjh, wcs, src["ra"], src["dec"], size)
        vis_im = cutout[0]  # VIS (I)
        y_im = cutout[1] if needs_y else None   # NIR-Y
        j_im = cutout[2] if needs_j else None   # NIR-J

        for variant, out_dir in out_dirs.items():
            out_path = os.path.join(out_dir, f"{stem}.{fmt}")
            if os.path.exists(out_path):
                continue
            try:
                rgb = _render_bulk_variant(variant, vis_im, y_im, j_im)
                if rgb is None:
                    continue
                Image.fromarray(rgb).save(out_path, **save_kw)
                counts[variant] += 1
            except Exception:
                logging.exception("  bulk_euclid %s failed for %s", variant, src["id"])

    return counts


def _render_bulk_variant(variant: str, vis: np.ndarray,
                         y: np.ndarray | None,
                         j: np.ndarray | None) -> np.ndarray | None:
    """Dispatch to the correct bulk_euclid rendering function.

    bulk_euclid functions preserve FITS row order (origin = bottom-left),
    so we flipud the result to put north up, matching azulero/eummy.
    """
    rgb = None

    if variant == "gz_arcsinh_vis_y":
        rgb = make_composite_cutout(vis.copy(), y.copy(), vis_q=100, nisp_q=0.2)

    elif variant == "gz_arcsinh_vis_only":
        grey = make_vis_only_cutout(vis.copy(), q=100)
        rgb = np.stack([grey, grey, grey], axis=2)

    elif variant == "gz_arcsinh_triple":
        rgb = make_triple_cutout(vis.copy(), y.copy(), j.copy(),
                                 short_q=100, mid_q=0.2, long_q=0.1)

    elif variant == "sw_mtf_vis_only":
        vis_mtf = apply_MTF(vis.copy())
        rgb = np.stack([vis_mtf, vis_mtf, vis_mtf], axis=2)

    elif variant == "sw_mtf_vis_y":
        vis_mtf = apply_MTF(vis.copy())
        y_mtf = apply_MTF(y.copy())
        mean_mtf = np.mean([vis_mtf, y_mtf], axis=0).astype(np.uint8)
        rgb = np.stack([y_mtf, mean_mtf, vis_mtf], axis=2)
        rgb = replace_luminosity_channel(rgb, rgb_channel_for_luminosity=2,
                                         desaturate_speckles=False)

    elif variant == "sw_mtf_vis_y_j":
        vis_mtf = apply_MTF(vis.copy())
        y_mtf = apply_MTF(y.copy())
        j_mtf = apply_MTF(j.copy())
        rgb = np.stack([j_mtf, y_mtf, vis_mtf], axis=2)
        rgb = replace_luminosity_channel(rgb, rgb_channel_for_luminosity=2,
                                         desaturate_speckles=False)

    else:
        logging.warning("Unknown bulk_euclid variant: %s", variant)
        return None

    return np.flipud(rgb)


# ── FITS-input mode helpers ────────────────────────────────────────────────

_IYJH_BAND_INDEX = {"VIS": 0, "NIR_Y": 1, "NIR_J": 2, "NIR_H": 3}


def _load_fits_cutout(path: str) -> np.ndarray | None:
    """Load a multi-extension FITS cutout into a (4, H, W) IYJH array.

    Extensions are mapped to bands via FITS_BAND_ORDER. Missing bands are
    zero-filled. Returns None if no usable image extensions found.
    """
    with fits.open(path) as hdul:
        img_hdus = [h for h in hdul if h.data is not None and h.data.ndim == 2]
        if not img_hdus:
            return None

        h, w = img_hdus[0].data.shape
        iyjh = np.zeros((4, h, w), dtype=np.float32)

        for i, hdu in enumerate(img_hdus):
            if i >= len(FITS_BAND_ORDER):
                break
            band_name = FITS_BAND_ORDER[i]
            idx = _IYJH_BAND_INDEX.get(band_name)
            if idx is not None:
                iyjh[idx] = hdu.data.astype(np.float32)

    return iyjh


def _render_single_fits(args: tuple) -> tuple[str, int, dict[str, int]]:
    """Render colour images for a single FITS cutout file.

    Returns (stem, n_azulero, {variant: count}).
    """
    fits_path, azulero_out, bulk_out_dirs = args

    stem = os.path.splitext(os.path.basename(fits_path))[0]

    iyjh = _load_fits_cutout(fits_path)
    if iyjh is None:
        logging.warning("No image data in %s — skipping", fits_path)
        return stem, 0, {v: 0 for v in bulk_out_dirs}

    az_fmt = AZULERO_FORMAT.lower()
    be_fmt = BULK_EUCLID_FORMAT.lower()
    n_azulero = 0
    bulk_counts: dict[str, int] = {v: 0 for v in bulk_out_dirs}

    if ENABLE_AZULERO:
        out_path = os.path.join(azulero_out, f"{stem}.{az_fmt}")
        if not os.path.exists(out_path):
            try:
                transform = build_azulero_transform()
                rgb_f = _render_azulero_rgb(iyjh, transform)
                rgb = (np.clip(rgb_f, 0.0, 1.0) * 255).round().astype(np.uint8)
                save_kw = {"quality": JPEG_QUALITY} if az_fmt == "jpg" else {}
                Image.fromarray(rgb).save(out_path, **save_kw)
                n_azulero = 1
            except Exception:
                logging.exception("  azulero failed for %s", stem)

    if ENABLE_BULK_EUCLID and bulk_out_dirs:
        import cv2
        cv2.setNumThreads(1)
        vis_im = iyjh[0]
        y_im = iyjh[1]
        j_im = iyjh[2]
        for variant, out_dir in bulk_out_dirs.items():
            out_path = os.path.join(out_dir, f"{stem}.{be_fmt}")
            if os.path.exists(out_path):
                continue
            try:
                rgb = _render_bulk_variant(variant, vis_im, y_im, j_im)
                if rgb is None:
                    continue
                save_kw = {"quality": JPEG_QUALITY} if be_fmt == "jpg" else {}
                Image.fromarray(rgb).save(out_path, **save_kw)
                bulk_counts[variant] += 1
            except Exception:
                logging.exception("  bulk_euclid %s failed for %s", variant, stem)

    return stem, n_azulero, bulk_counts


def render_fits_dir():
    """Colour-render pre-made FITS cutout files from INPUT_FITS_DIR."""
    fits_files = sorted(glob.glob(os.path.join(INPUT_FITS_DIR, "*.fits")))
    if not fits_files:
        logging.error("No .fits files found in %s", INPUT_FITS_DIR)
        return
    logging.info("FITS-input mode: %d files in %s", len(fits_files), INPUT_FITS_DIR)

    if ENABLE_EUMMY:
        logging.warning("Eummy is not supported in FITS-input mode (needs tile-level FITS) — skipping")

    azulero_out = os.path.join(OUTPUT_DIR, "azulero")
    if ENABLE_AZULERO:
        os.makedirs(azulero_out, exist_ok=True)

    bulk_out_dirs: dict[str, str] = {}
    if ENABLE_BULK_EUCLID:
        for variant in BULK_EUCLID_OUTPUTS:
            d = os.path.join(OUTPUT_DIR, variant)
            os.makedirs(d, exist_ok=True)
            bulk_out_dirs[variant] = d

    active = []
    if ENABLE_AZULERO:
        active.append("azulero")
    if bulk_out_dirs:
        active.extend(BULK_EUCLID_OUTPUTS)
    logging.info("Active renderers: %s", ", ".join(active))
    logging.info("Band mapping: %s", " → ".join(
        f"CHANNEL_{i+1}={b}" for i, b in enumerate(FITS_BAND_ORDER)))

    work_items = [(fp, azulero_out, bulk_out_dirs) for fp in fits_files]

    total_az = 0
    total_bulk: dict[str, int] = {v: 0 for v in bulk_out_dirs}
    done = 0
    n_total = len(work_items)

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(_render_single_fits, work_items)
        if PROGRESS_BAR:
            from tqdm import tqdm
            results_iter = tqdm(results_iter, total=n_total, unit="file",
                                desc="Rendering FITS cutouts")
        for stem, n_az, bc in results_iter:
            total_az += n_az
            for v, c in bc.items():
                total_bulk[v] = total_bulk.get(v, 0) + c
            done += 1
            if PROGRESS_BAR:
                bulk_str = "  ".join(f"{v}={total_bulk[v]}" for v in sorted(total_bulk))
                results_iter.set_postfix_str(f"az={total_az} {bulk_str}")
            elif done % 50 == 0 or done == n_total:
                bulk_str = "  ".join(f"{v}={total_bulk[v]}" for v in sorted(total_bulk))
                logging.info("[%d/%d files] azulero=%d  %s",
                             done, n_total, total_az, bulk_str)

    lines = [f"Done. {done} files processed"]
    if ENABLE_AZULERO:
        lines.append(f"  azulero={total_az} → {azulero_out}/")
    for v, d in bulk_out_dirs.items():
        lines.append(f"  {v}={total_bulk.get(v, 0)} → {d}/")
    logging.info("\n".join(lines))


def run_eummy_tile(iyjh_paths: list[str], sources: list[dict],
                   tile_dir: str, eummy_out_dir: str) -> int:
    """Run eummy for one tile and rename outputs to {id}.png. Returns count produced."""
    cutout_arcsec = max(
        s.get("size_pixel", DEFAULT_CUTOUT_PIXELS) for s in sources
    ) * VIS_ARCSEC_PER_PX

    _setup_tile_workspace(tile_dir, iyjh_paths)
    catalog_path = os.path.join(tile_dir, "cutout_catalog.fits")
    _write_tile_catalog(sources, catalog_path)

    cmd = [
        _find_eummy_exe(),
        "--path", tile_dir,
        "--cutouts", catalog_path, f'{cutout_arcsec}"',
        "--nthreads", str(N_WORKERS),
    ]
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return rename_eummy_cutouts(tile_dir, sources, eummy_out_dir)


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def process_tile(args: tuple) -> tuple[int, int, int, int, dict[str, int], int]:
    """Per-tile worker. Returns (tile_id, n_azulero, n_stci, n_eummy, bulk_counts, n_skipped)."""
    tile_id, sources, azulero_out, stci_out, eummy_out, bulk_out_dirs = args

    # Skip tiles already fully processed.
    az_fmt = AZULERO_FORMAT.lower()
    em_fmt = EUMMY_FORMAT.lower()
    be_fmt = BULK_EUCLID_FORMAT.lower()
    stems = [_make_stem(s) for s in sources]
    az_exists = sum(1 for st in stems if os.path.exists(os.path.join(azulero_out, f"{st}.{az_fmt}")))
    em_exists = sum(1 for st in stems if os.path.exists(os.path.join(eummy_out, f"{st}.{em_fmt}")))
    be_exists = all(
        all(os.path.exists(os.path.join(d, f"{st}.{be_fmt}")) for st in stems)
        for d in bulk_out_dirs.values()
    ) if bulk_out_dirs else True
    if az_exists > 0 and em_exists == len(sources) and be_exists:
        return tile_id, 0, 0, 0, {v: 0 for v in bulk_out_dirs}, 0

    ra0  = sources[0]["ra"]
    dec0 = sources[0]["dec"]
    release_dir = sources[0].get("release_dir")

    iyjh_paths = resolve_iyjh_paths(tile_id, ra0, dec0, release_dir)
    if iyjh_paths is None:
        logging.warning("Tile %s: FITS not found — skipping %d sources",
                        tile_id, len(sources))
        return tile_id, 0, 0, 0, {v: 0 for v in bulk_out_dirs}, len(sources)

    # ── Load tile data ──────────────────────────────────────────────────────
    handles = [fits.open(p, memmap=True) for p in iyjh_paths]
    try:
        iyjh = np.stack([h[0].data.astype(np.float32) for h in handles])
        wcs  = WCS(handles[0][0].header)
    finally:
        for h in handles:
            h.close()

    # ── Azulero pass ────────────────────────────────────────────────────────
    n_azulero = 0
    if ENABLE_AZULERO:
        n_azulero = render_azulero_tile(iyjh, wcs, sources, azulero_out)

    # ── STCI pass ──────────────────────────────────────────────────────────
    n_stci = 0
    if ENABLE_STCI:
        n_stci = render_stci_tile(iyjh, wcs, sources, stci_out)

    # ── bulk_euclid pass (GZ + SW) ──────────────────────────────────────────
    bulk_counts = {v: 0 for v in bulk_out_dirs}
    if ENABLE_BULK_EUCLID and bulk_out_dirs:
        bulk_counts = render_bulk_euclid_tile(iyjh, wcs, sources, bulk_out_dirs)

    del iyjh

    # ── Eummy pass ──────────────────────────────────────────────────────────
    n_eummy = 0
    if ENABLE_EUMMY:
        em_tile_dir = os.path.join(eummy_out, f"_tile_ws_{tile_id}")
        try:
            n_eummy = run_eummy_tile(iyjh_paths, sources, em_tile_dir, eummy_out)
        except subprocess.CalledProcessError:
            logging.exception("Tile %s: eummy failed", tile_id)
            n_eummy = 0
        finally:
            shutil.rmtree(em_tile_dir, ignore_errors=True)

    return tile_id, n_azulero, n_stci, n_eummy, bulk_counts, 0


def main():
    if INPUT_FITS_DIR:
        render_fits_dir()
        return

    logging.info("Loading sources from %s", SOURCES_CSV)
    df = load_sources(SOURCES_CSV)
    logging.info("%d sources across %d tiles", len(df), df["tile_index"].nunique())

    azulero_out = os.path.join(OUTPUT_DIR, "azulero")
    stci_out    = os.path.join(OUTPUT_DIR, "stci")
    eummy_out   = os.path.join(OUTPUT_DIR, "eummy")
    if ENABLE_AZULERO:
        os.makedirs(azulero_out, exist_ok=True)
    if ENABLE_STCI:
        os.makedirs(stci_out, exist_ok=True)
    if ENABLE_EUMMY:
        os.makedirs(eummy_out, exist_ok=True)

    bulk_out_dirs: dict[str, str] = {}
    if ENABLE_BULK_EUCLID:
        for variant in BULK_EUCLID_OUTPUTS:
            d = os.path.join(OUTPUT_DIR, variant)
            os.makedirs(d, exist_ok=True)
            bulk_out_dirs[variant] = d

    active = []
    if ENABLE_AZULERO:
        active.append("azulero")
    if ENABLE_STCI:
        active.append("stci")
    if ENABLE_EUMMY:
        active.append("eummy")
    if bulk_out_dirs:
        active.extend(BULK_EUCLID_OUTPUTS)
    logging.info("Active renderers: %s", ", ".join(active))

    work_items = [
        (int(tile_id),
         group[[c for c in ["id", "object_id", "tile_index", "ra", "dec", "size_pixel", "release_dir"]
                if c in group.columns]].to_dict("records"),
         azulero_out,
         stci_out,
         eummy_out,
         bulk_out_dirs)
        for tile_id, group in df.groupby("tile_index")
    ]

    n_tiles = len(work_items)
    total_az = total_stci = total_em = total_skip = done = 0
    total_bulk: dict[str, int] = {v: 0 for v in bulk_out_dirs}

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(process_tile, work_items)
        if PROGRESS_BAR:
            from tqdm import tqdm
            results_iter = tqdm(results_iter, total=n_tiles, unit="tile",
                                desc="Rendering cutouts")
        for tile_id, n_az, n_st, n_em, bc, n_skip in results_iter:
            total_az   += n_az
            total_stci += n_st
            total_em   += n_em
            total_skip += n_skip
            for v, c in bc.items():
                total_bulk[v] = total_bulk.get(v, 0) + c
            done += 1
            if PROGRESS_BAR:
                bulk_str = "  ".join(f"{v}={total_bulk[v]}" for v in sorted(total_bulk))
                results_iter.set_postfix_str(
                    f"az={total_az} stci={total_stci} em={total_em} {bulk_str} skip={total_skip}")
            elif done % 50 == 0 or done == n_tiles:
                bulk_str = "  ".join(f"{v}={total_bulk[v]}" for v in sorted(total_bulk))
                logging.info("[%d/%d tiles] azulero=%d  stci=%d  eummy=%d  %s  skipped=%d",
                             done, n_tiles, total_az, total_stci, total_em, bulk_str, total_skip)

    lines = [f"Done. skipped={total_skip}"]
    if ENABLE_AZULERO:
        lines.append(f"  azulero={total_az} → {azulero_out}/")
    if ENABLE_STCI:
        lines.append(f"  stci={total_stci} → {stci_out}/")
    if ENABLE_EUMMY:
        lines.append(f"  eummy={total_em} → {eummy_out}/")
    for v, d in bulk_out_dirs.items():
        lines.append(f"  {v}={total_bulk.get(v, 0)} → {d}/")
    logging.info("\n".join(lines))


if __name__ == "__main__":
    main()
