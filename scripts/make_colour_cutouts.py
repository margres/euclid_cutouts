"""
Produce azulero JPEG + eummy PNG colour cutouts for Euclid sources.

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

_CUTANA_ROOT = "/media/user/astronomaly-euclid"
if _CUTANA_ROOT not in sys.path:
    sys.path.insert(0, _CUTANA_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from fits_path_utils import find_fits_paths_any_release, find_fits_paths  # noqa: E402
from cutana_datalabs import azulero_render  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Input CSV.
# Required columns : ra, dec
# Optional columns :
#   id              — used as filename stem when NAMING="id" (defaults to row index)
#   object_id       — MER catalog object ID; required when NAMING="q1_slde"
#   tile_index      — Euclid tile ID (auto-looked-up from TILE_CENTRES_CSV if absent)
#   size_pixel      — cutout size in VIS pixels, per source
#   size_arcsec     — cutout size in arcsec, per source (size_pixel takes precedence)
SOURCES_CSV           = "sources.csv"

# HEALPix tile map — used to resolve tile_index from ra/dec when absent from the CSV.
# tile_centres.csv is used only to enrich tile_index with release_dir.
HEALPIX_MAP      = "/media/home/my_workspace/cutana_dr1_pipeline/data/tile_index_map.v1.2.fits.gz"
TILE_CENTRES_CSV = "/media/home/my_workspace/cutana_dr1_pipeline/tile_centres.csv"

# Output root directory. All outputs share this root, in subfolders named
# after the format/stretch:
#   {OUTPUT_DIR}/fits/     — FITS cutouts (Cutana)
#   {OUTPUT_DIR}/azulero/  — colour JPEGs (azulero stretch)
#   {OUTPUT_DIR}/eummy/    — colour PNGs  (eummy stretch)
# All three are flat (no per-tile subdirs), mirroring Cutana's default structure.
OUTPUT_DIR            = "cutouts"

# File naming convention for output cutouts.
#   "id"            — use the id column (or row index if absent)
#   "q1_slde"       — {tile_index}_{object_id}  (requires object_id column)
#   "cutana_default"— {id}_{ra:.6f}_{dec:.6f}   (mirrors Cutana's built-in template)
NAMING                = "id"

# Default cutout size when not specified per source
DEFAULT_CUTOUT_PIXELS = 101
VIS_ARCSEC_PER_PX     = 0.1   # VIS plate scale (arcsec per pixel)

N_WORKERS             = 1

# Fallback release dirs — searched in order when a tile is not in TILE_CENTRES_CSV.
# Q1 (R1 only):
RELEASE_DIRS          = ["/media/home/data/euclid_q1/Q1_R1"]
# DR1 (comment Q1 line above and uncomment below):
# RELEASE_DIRS        = ["/media/home/data/euclid_idr1/DR1/R2",
#                        "/media/home/data/euclid_idr1/DR1/R1"]

BANDS_3               = ["VIS", "NIR_Y", "NIR_J"]  # NIR_H derived by azulero_render.find_iyjh_paths
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
    """Rename RA/Dec columns to 'ra' and 'dec', case-insensitive."""
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
    return azulero_render.find_iyjh_paths(paths_3)


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
    """Match eummy PNGs to sources by RA/Dec and move to eummy_out_dir/{id}.png."""
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

        dest = os.path.join(eummy_out_dir, f"{_make_stem(sources[i])}.png")
        if os.path.exists(dest):
            continue
        os.replace(png_path, dest)
        n_renamed += 1
    return n_renamed


def render_azulero_tile(iyjh: np.ndarray, wcs: WCS,
                        sources: list[dict], out_dir: str) -> int:
    """Render azulero JPEGs for all sources in one tile. Returns number written."""
    transform = azulero_render.build_transform()
    n_ok = n_fail = 0
    for src in sources:
        out_path = os.path.join(out_dir, f"{_make_stem(src)}.jpg")
        if os.path.exists(out_path):
            continue
        try:
            size   = int(src.get("size_pixel", DEFAULT_CUTOUT_PIXELS))
            cutout = _extract_cutout(iyjh, wcs, src["ra"], src["dec"], size)
            rgb    = azulero_render.render_rgb_uint8(cutout, transform)
            Image.fromarray(rgb).save(out_path, format="JPEG", quality=95)
            n_ok += 1
        except Exception:
            logging.exception("  azulero render failed for %s", src["id"])
            n_fail += 1

    if n_fail:
        logging.warning("  %d renders failed in %s", n_fail, out_dir)
    return n_ok


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


def process_tile(args: tuple) -> tuple[int, int, int, int]:
    """Per-tile worker. Returns (tile_id, n_azulero_ok, n_eummy_ok, n_skipped)."""
    tile_id, sources, azulero_out, eummy_out = args

    ra0  = sources[0]["ra"]
    dec0 = sources[0]["dec"]
    release_dir = sources[0].get("release_dir")

    iyjh_paths = resolve_iyjh_paths(tile_id, ra0, dec0, release_dir)
    if iyjh_paths is None:
        logging.warning("Tile %s: FITS not found — skipping %d sources",
                        tile_id, len(sources))
        return tile_id, 0, 0, len(sources)

    # ── Azulero pass ────────────────────────────────────────────────────────
    handles = [fits.open(p, memmap=True) for p in iyjh_paths]
    try:
        iyjh = np.stack([h[0].data.astype(np.float32) for h in handles])
        wcs  = WCS(handles[0][0].header)
    finally:
        for h in handles:
            h.close()

    n_azulero = render_azulero_tile(iyjh, wcs, sources, azulero_out)
    del iyjh

    # ── Eummy pass ──────────────────────────────────────────────────────────
    em_tile_dir = os.path.join(eummy_out, f"_tile_ws_{tile_id}")
    try:
        n_eummy = run_eummy_tile(iyjh_paths, sources, em_tile_dir, eummy_out)
    except subprocess.CalledProcessError:
        logging.exception("Tile %s: eummy failed", tile_id)
        n_eummy = 0
    finally:
        shutil.rmtree(em_tile_dir, ignore_errors=True)

    return tile_id, n_azulero, n_eummy, 0


def main():
    logging.info("Loading sources from %s", SOURCES_CSV)
    df = load_sources(SOURCES_CSV)
    logging.info("%d sources across %d tiles", len(df), df["tile_index"].nunique())

    azulero_out = os.path.join(OUTPUT_DIR, "azulero")
    eummy_out   = os.path.join(OUTPUT_DIR, "eummy")
    os.makedirs(azulero_out, exist_ok=True)
    os.makedirs(eummy_out,   exist_ok=True)

    work_items = [
        (int(tile_id),
         group[[c for c in ["id", "object_id", "tile_index", "ra", "dec", "size_pixel", "release_dir"]
                if c in group.columns]].to_dict("records"),
         azulero_out,
         eummy_out)
        for tile_id, group in df.groupby("tile_index")
    ]

    n_tiles = len(work_items)
    total_az = total_em = total_skip = done = 0

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for tile_id, n_az, n_em, n_skip in pool.imap_unordered(process_tile, work_items):
            total_az   += n_az
            total_em   += n_em
            total_skip += n_skip
            done       += 1
            if done % 50 == 0 or done == n_tiles:
                logging.info("[%d/%d tiles] azulero=%d  eummy=%d  skipped=%d",
                             done, n_tiles, total_az, total_em, total_skip)

    logging.info(
        "Done. azulero JPEGs: %d  eummy PNGs: %d  skipped: %d\n"
        "  azulero → %s/\n"
        "  eummy   → %s/",
        total_az, total_em, total_skip, azulero_out, eummy_out,
    )


if __name__ == "__main__":
    main()
