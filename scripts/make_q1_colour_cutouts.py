"""
Produce azulero JPEG + eummy PNG colour cutouts for Euclid BYOL sources.

Works with Q1 and DR1 data — set the CONFIG block below for the target release.
Processes one tile at a time via multiprocessing. Each tile: resolves FITS paths,
renders azulero JPEGs (in-memory, per-source), runs eummy CLI (whole tile), renames.

Run:
    python make_q1_colour_cutouts.py
"""

import glob
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

# Make cutana_datalabs and local scripts importable
_CUTANA_ROOT = "/media/user/astronomaly-euclid"
if _CUTANA_ROOT not in sys.path:
    sys.path.insert(0, _CUTANA_ROOT)

sys.path.insert(0, os.path.dirname(__file__))

from fits_path_utils import find_fits_paths_any_release  # noqa: E402
from cutana_datalabs import azulero_render  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Source list: parquet index gives source IDs; coords CSV provides RA/Dec.
INPUT_PARQUET    = "/media/user/astronomaly-euclid/q1_SL_data/features_pca_97_swin_mtf_vis_y_j_200k.parquet"
COORDS_CSV       = "/media/user/search_engine_catalogue/almost_full_q1.csv"

# Column names in COORDS_CSV (lowercase; adjust if your DR1 coords CSV differs)
COORDS_ID_COL    = "object_id"
COORDS_RA_COL    = "right_ascension"
COORDS_DEC_COL   = "declination"

# Output directories
AZULERO_OUT      = "/media/user/euclid_cutouts/results/q1_colour/azulero"
EUMMY_OUT        = "/media/user/euclid_cutouts/results/q1_colour/eummy"

CUTOUT_PIXELS    = 101
CUTOUT_ARCSEC    = 10.1
N_WORKERS        = 1

# Output image format: "jpg" (lossy, ~20 KB/cutout) or "png" (lossless, ~60 KB/cutout)
AZULERO_FORMAT   = "jpg"
EUMMY_FORMAT     = "jpg"
JPEG_QUALITY     = 99

# Release dirs, tried in order — first complete set of FITS wins (put R2 before R1).
# Q1 (R1 only):
RELEASE_DIRS     = ["/media/home/data/euclid_q1/Q1_R1"]
# DR1 (R2 first, fall back to R1):
# RELEASE_DIRS   = ["/media/home/data/euclid_idr1/DR1/R2", "/media/home/data/euclid_idr1/DR1/R1"]

BANDS_3          = ["VIS", "NIR_Y", "NIR_J"]   # NIR_H derived by azulero_render.find_iyjh_paths
BAND_TO_INST     = {"VIS": "VIS", "NIR_Y": "NISP", "NIR_J": "NISP"}
# ── END CONFIG ────────────────────────────────────────────────────────────────


def parse_source_id(source_id: str) -> tuple[int, int]:
    """'{tileID}_{objectID}' (NEG-encoded) -> (tile_id, object_id)."""
    parts = str(source_id).split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected source_id format: {source_id!r}")
    tile_str, obj_str = parts
    obj_id = -int(obj_str[3:]) if obj_str.startswith("NEG") else int(obj_str)
    return int(tile_str), obj_id


def load_sources(parquet_path: str, coords_csv: str) -> pd.DataFrame:
    """Load 1M SourceIDs from parquet, join RA/Dec from coords_csv.

    Returns DataFrame with columns: source_id, tile_id, ra, dec.
    Drops sources with no RA/Dec match.
    """
    ids = pd.read_parquet(parquet_path, columns=[]).index.astype(str)
    parsed = [parse_source_id(s) for s in ids]
    df = pd.DataFrame({
        "source_id": list(ids),
        "tile_id": [t for t, _ in parsed],
        "object_id": [o for _, o in parsed],
    })

    coords = pd.read_csv(coords_csv, usecols=[COORDS_ID_COL, COORDS_RA_COL, COORDS_DEC_COL])
    # object_id is globally unique in the Euclid MER catalog, so joining on
    # object_id alone (without tile_id) is safe — each object_id maps to exactly
    # one sky position regardless of which tile it was detected in.
    df = df.merge(coords, left_on="object_id", right_on=COORDS_ID_COL, how="left").rename(
        columns={COORDS_RA_COL: "ra", COORDS_DEC_COL: "dec"}
    )
    n_before = len(df)
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    if n_before > len(df):
        logging.warning("%d sources had no RA/Dec match — dropped", n_before - len(df))
    return df[["source_id", "tile_id", "ra", "dec"]]


def resolve_iyjh_paths(tile_id: int, ra: float, dec: float) -> list[str] | None:
    """Resolve 4 IYJH FITS paths for tile_id from Q1_RELEASE_DIRS.

    Uses find_fits_paths_any_release for VIS/NIR_Y/NIR_J, then
    azulero_render.find_iyjh_paths to glob NIR_H.
    Returns None if any band is missing.
    """
    paths_3 = find_fits_paths_any_release(
        tile_id, BANDS_3, RELEASE_DIRS, BAND_TO_INST, ra=ra, dec=dec
    )
    if paths_3 is None:
        return None
    return azulero_render.find_iyjh_paths(paths_3)


def _extract_cutout(iyjh: np.ndarray, wcs: WCS, ra: float, dec: float, size: int) -> np.ndarray:
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
        out[:, y0c - y0:y0c - y0 + (y1c - y0c), x0c - x0:x0c - x0 + (x1c - x0c)] = \
            iyjh[:, y0c:y1c, x0c:x1c]
    return out


# Matches the "TILE{id}_{RA}{+/-}{|Dec|}.png" filenames eummy writes for cutouts
_EUMMY_RE = re.compile(r"^.+_(-?\d+\.\d+)([+-]\d+\.\d+)\.png$")


def _find_eummy_exe() -> str:
    """Locate the eummy console script, falling back to the directory of the
    current Python interpreter (handles envs where it isn't on PATH)."""
    exe = shutil.which("eummy")
    if exe:
        return exe
    candidate = os.path.join(os.path.dirname(sys.executable), "eummy")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "Could not find the 'eummy' executable. Install it with "
        "'pip install eummy' in this environment."
    )


def _setup_tile_workspace(tile_dir: str, iyjh_paths: list[str]) -> None:
    """Symlink the FITS files for a tile into tile_dir, preserving their basenames."""
    os.makedirs(tile_dir, exist_ok=True)
    for path in iyjh_paths:
        link = os.path.join(tile_dir, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(path, link)


def _write_tile_catalog(sources: list[dict], catalog_path: str) -> None:
    """Write a small FITS binary table with RA/DEC columns for eummy."""
    ra_arr = np.array([s["ra"] for s in sources], dtype=np.float64)
    dec_arr = np.array([s["dec"] for s in sources], dtype=np.float64)
    cols = [
        fits.Column(name="RA", format="D", array=ra_arr),
        fits.Column(name="DEC", format="D", array=dec_arr),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(catalog_path, overwrite=True)


def rename_eummy_cutouts(tile_dir: str, sources: list[dict], eummy_out_dir: str,
                         match_tol_deg: float = 1.5 / 3600) -> int:
    """Match eummy-written PNGs to sources by RA/Dec and move to eummy_out_dir/{source_id}.png.

    Returns the number of files renamed/moved.
    """
    ra_arr = np.array([s["ra"] for s in sources])
    dec_arr = np.array([s["dec"] for s in sources])

    os.makedirs(eummy_out_dir, exist_ok=True)
    n_renamed = 0
    for png_path in glob.glob(os.path.join(tile_dir, "TILE*_*.png")):
        fname = os.path.basename(png_path)
        match = _EUMMY_RE.match(fname)
        if not match:
            continue
        ra, dec = float(match.group(1)), float(match.group(2))

        dra = (ra_arr - ra) * np.cos(np.radians(dec))
        ddec = dec_arr - dec
        dist = np.hypot(dra, ddec)
        i = int(np.argmin(dist))
        if dist[i] > match_tol_deg:
            logging.warning(
                "  No source within match tolerance for %s (closest %.2f\")",
                fname, dist[i] * 3600,
            )
            continue

        source_id = sources[i]["source_id"]
        fmt = EUMMY_FORMAT.lower()
        dest = os.path.join(eummy_out_dir, f"{source_id}.{fmt}")
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
    """Render azulero JPEGs for all sources in one tile.

    iyjh: (4, H, W) float32 — full tile already loaded in memory.
    wcs:  WCS from band-0 header.
    sources: list of dicts with keys source_id, ra, dec.
    out_dir: tile-level directory (already created by caller).
    Returns number of JPEGs written.
    """
    fmt = AZULERO_FORMAT.lower()
    transform = azulero_render.build_transform()
    n_ok = 0
    n_fail = 0
    for src in sources:
        out_path = os.path.join(out_dir, f"{src['source_id']}.{fmt}")
        if os.path.exists(out_path):
            continue
        try:
            cutout = _extract_cutout(iyjh, wcs, src["ra"], src["dec"], CUTOUT_PIXELS)
            rgb = azulero_render.render_rgb_uint8(cutout, transform)
            save_kw = {"quality": JPEG_QUALITY} if fmt == "jpg" else {}
            Image.fromarray(rgb).save(out_path, **save_kw)
            n_ok += 1
        except Exception:
            logging.exception("  azulero render failed for %s", src['source_id'])
            n_fail += 1

    if n_fail:
        logging.warning("  %d renders failed in %s", n_fail, out_dir)
    return n_ok


def run_eummy_tile(iyjh_paths: list[str], sources: list[dict],
                   tile_dir: str, eummy_out_dir: str) -> int:
    """Run eummy for one tile and rename the output PNGs.

    Sets up a workspace with symlinked FITS files, writes a FITS catalog,
    runs eummy, then renames the cutouts to {source_id}.png in eummy_out_dir.
    Returns number of cutouts produced.
    """
    _setup_tile_workspace(tile_dir, iyjh_paths)
    catalog_path = os.path.join(tile_dir, "cutout_catalog.fits")
    _write_tile_catalog(sources, catalog_path)
    cmd = [
        _find_eummy_exe(),
        "--path", tile_dir,
        "--cutouts", catalog_path, f'{CUTOUT_ARCSEC}"',
        "--nthreads", str(N_WORKERS),
    ]
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return rename_eummy_cutouts(tile_dir, sources, eummy_out_dir)


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def process_tile(args: tuple) -> tuple[int, int, int, int]:
    """Per-tile worker. Args: (tile_id, sources, azulero_out, eummy_out).
    Returns: (tile_id, n_azulero_ok, n_eummy_ok, n_skipped).
    """
    tile_id, sources, azulero_out, eummy_out = args

    # Check each pass independently so we can skip one without re-running the other.
    az_tile_dir = os.path.join(azulero_out, str(tile_id))
    az_done = os.path.isdir(az_tile_dir) and len(os.listdir(az_tile_dir)) > 0
    em_done = all(os.path.exists(os.path.join(eummy_out, f"{s['source_id']}.{EUMMY_FORMAT.lower()}")) for s in sources)
    if az_done and em_done:
        return tile_id, 0, 0, 0

    # Resolve FITS paths using first source's RA/Dec for coverage check
    ra0, dec0 = sources[0]["ra"], sources[0]["dec"]
    iyjh_paths = resolve_iyjh_paths(tile_id, ra0, dec0)
    if iyjh_paths is None:
        logging.warning("Tile %s: FITS not found — skipping %d sources", tile_id, len(sources))
        return tile_id, 0, 0, len(sources)

    # ── Azulero pass ────────────────────────────────────────────────────────
    n_azulero = 0
    if not az_done:
        os.makedirs(az_tile_dir, exist_ok=True)
        handles = [fits.open(p, memmap=True) for p in iyjh_paths]
        try:
            iyjh = np.stack([h[0].data.astype(np.float32) for h in handles])
            wcs  = WCS(handles[0][0].header)
        finally:
            for h in handles:
                h.close()
        n_azulero = render_azulero_tile(iyjh, wcs, sources, az_tile_dir)
        del iyjh

    # ── Eummy pass ──────────────────────────────────────────────────────────
    n_eummy = 0
    if not em_done:
        em_tile_dir = os.path.join(eummy_out, f"_tile_ws_{tile_id}")
        try:
            n_eummy = run_eummy_tile(iyjh_paths, sources, em_tile_dir, eummy_out)
        except subprocess.CalledProcessError:
            logging.exception("Tile %s: eummy failed", tile_id)
        finally:
            shutil.rmtree(em_tile_dir, ignore_errors=True)

    return tile_id, n_azulero, n_eummy, 0


def main():
    logging.info("Loading sources from %s", INPUT_PARQUET)
    df = load_sources(INPUT_PARQUET, COORDS_CSV)
    logging.info("%d sources across %d tiles", len(df), df["tile_id"].nunique())

    os.makedirs(AZULERO_OUT, exist_ok=True)
    os.makedirs(EUMMY_OUT,   exist_ok=True)

    work_items = [
        (int(tile_id),
         group[["source_id", "ra", "dec"]].to_dict("records"),
         AZULERO_OUT,
         EUMMY_OUT)
        for tile_id, group in df.groupby("tile_id")
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
                logging.info(
                    "[%d/%d tiles] azulero=%d eummy=%d skipped=%d",
                    done, n_tiles, total_az, total_em, total_skip,
                )

    logging.info(
        "Done. azulero JPEGs: %d  eummy PNGs: %d  skipped: %d\n"
        "  azulero → %s/{tileID}/\n"
        "  eummy   → %s/",
        total_az, total_em, total_skip, AZULERO_OUT, EUMMY_OUT,
    )


if __name__ == "__main__":
    main()
