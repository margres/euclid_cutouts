"""
Produce azulero JPEG + eummy PNG colour cutouts for all ~1.08M Q1 BYOL sources.

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
INPUT_PARQUET    = "/media/user/astronomaly-euclid/q1_SL_data/features_pca_97_swin_mtf_vis_y_j_200k.parquet"
COORDS_CSV       = "/media/user/search_engine_catalogue/almost_full_q1.csv"
AZULERO_OUT      = "/media/user/cutana_dr1_pipeline/results/q1_colour/azulero"
EUMMY_OUT        = "/media/user/cutana_dr1_pipeline/results/q1_colour/eummy"
CUTOUT_PIXELS    = 101
CUTOUT_ARCSEC    = 10.1
N_WORKERS        = max(1, os.cpu_count() // 2)
Q1_RELEASE_DIRS  = ["/media/home/data/euclid_q1/Q1_R1"]
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

    coords = pd.read_csv(coords_csv, usecols=["object_id", "right_ascension", "declination"])
    # object_id is globally unique in the Euclid MER catalog, so joining on
    # object_id alone (without tile_id) is safe — each object_id maps to exactly
    # one sky position regardless of which tile it was detected in.
    df = df.merge(coords, on="object_id", how="left").rename(
        columns={"right_ascension": "ra", "declination": "dec"}
    )
    n_before = len(df)
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    if n_before > len(df):
        logging.warning(f"{n_before - len(df)} sources had no RA/Dec match — dropped")
    return df[["source_id", "tile_id", "ra", "dec"]]


def resolve_iyjh_paths(tile_id: int, ra: float, dec: float) -> list[str] | None:
    """Resolve 4 IYJH FITS paths for tile_id from Q1_RELEASE_DIRS.

    Uses find_fits_paths_any_release for VIS/NIR_Y/NIR_J, then
    azulero_render.find_iyjh_paths to glob NIR_H.
    Returns None if any band is missing.
    """
    paths_3 = find_fits_paths_any_release(
        tile_id, BANDS_3, Q1_RELEASE_DIRS, BAND_TO_INST, ra=ra, dec=dec
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


def render_azulero_tile(iyjh: np.ndarray, wcs: WCS | None,
                        sources: list[dict], out_dir: str) -> int:
    """Render azulero JPEGs for all sources in one tile.

    iyjh: (4, H, W) float32 — full tile already loaded in memory.
    wcs:  WCS from band-0 header.
    sources: list of dicts with keys source_id, ra, dec.
    out_dir: tile-level directory (already created by caller).
    Returns number of JPEGs written.
    """
    transform = azulero_render.build_transform()
    n_ok = 0
    for src in sources:
        out_path = os.path.join(out_dir, f"{src['source_id']}.jpg")
        if os.path.exists(out_path):
            continue
        try:
            cutout = _extract_cutout(iyjh, wcs, src["ra"], src["dec"], CUTOUT_PIXELS)
            rgb = azulero_render.render_rgb_uint8(cutout, transform)
            Image.fromarray(rgb).save(out_path, format="JPEG", quality=95)
            n_ok += 1
        except Exception:
            logging.exception(f"  azulero render failed for {src['source_id']}")
    return n_ok
