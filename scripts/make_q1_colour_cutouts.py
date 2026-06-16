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

# azulero_render and find_fits_paths_any_release are imported lazily inside
# functions that need them so this module can be imported without azulero/cutana
# being installed (e.g. during unit tests of parse_source_id / load_sources).

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
    tile_str, obj_str = str(source_id).split("_", 1)
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
    df = df.merge(coords, on="object_id", how="left").rename(
        columns={"right_ascension": "ra", "declination": "dec"}
    )
    n_before = len(df)
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    if n_before - len(df):
        logging.warning(f"{n_before - len(df)} sources had no RA/Dec match — dropped")
    return df[["source_id", "tile_id", "ra", "dec"]]
