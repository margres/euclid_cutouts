"""
Produce STCI (PixInsight-like) colour cutouts for the Q1 BYOL source pool.

Tile-based: loads FITS once per tile, extracts per-source cutouts, runs the
STCI compose_pipeline (BN → CC → linked STF/HT → L-replacement → SCNR → saturation).

Uses the same source list and tile structure as make_q1_colour_cutouts.py.

Run:
    python make_q1_stci_cutouts.py
"""

import logging
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from fits_path_utils import find_fits_paths_any_release  # noqa: E402
from Tian_color import compose_pipeline, ComposeConfig, normalize_raw_channels_common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PARQUET    = "/media/user/astronomaly-euclid/q1_SL_data/features_pca_97_swin_mtf_vis_y_j_200k.parquet"
COORDS_CSV       = "/media/user/search_engine_catalogue/almost_full_q1.csv"
COORDS_ID_COL    = "object_id"
COORDS_RA_COL    = "right_ascension"
COORDS_DEC_COL   = "declination"

STCI_OUT         = "/media/user/euclid_cutouts/results/q1_colour/stci"
CUTOUT_PIXELS    = 101
N_WORKERS        = 1
OUTPUT_FORMAT    = "jpg"
JPEG_QUALITY     = 99

RELEASE_DIRS     = ["/media/home/data/euclid_q1/Q1_R1"]
BANDS_3          = ["VIS", "NIR_Y", "NIR_J"]
BAND_TO_INST     = {"VIS": "VIS", "NIR_Y": "NISP", "NIR_J": "NISP"}
STCI_CONFIG      = ComposeConfig()
# ── END CONFIG ────────────────────────────────────────────────────────────────


def parse_source_id(source_id: str) -> tuple[int, int]:
    parts = str(source_id).split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected source_id format: {source_id!r}")
    tile_str, obj_str = parts
    obj_id = -int(obj_str[3:]) if obj_str.startswith("NEG") else int(obj_str)
    return int(tile_str), obj_id


def load_sources(parquet_path: str, coords_csv: str) -> pd.DataFrame:
    ids = pd.read_parquet(parquet_path, columns=[]).index.astype(str)
    parsed = [parse_source_id(s) for s in ids]
    df = pd.DataFrame({
        "source_id": list(ids),
        "tile_id": [t for t, _ in parsed],
        "object_id": [o for _, o in parsed],
    })
    coords = pd.read_csv(coords_csv, usecols=[COORDS_ID_COL, COORDS_RA_COL, COORDS_DEC_COL])
    df = df.merge(coords, left_on="object_id", right_on=COORDS_ID_COL, how="left").rename(
        columns={COORDS_RA_COL: "ra", COORDS_DEC_COL: "dec"}
    )
    n_before = len(df)
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    if n_before > len(df):
        logging.warning("%d sources had no RA/Dec match — dropped", n_before - len(df))
    return df[["source_id", "tile_id", "ra", "dec"]]


def _extract_cutout(data_2d: np.ndarray, wcs: WCS, ra: float, dec: float, size: int) -> np.ndarray:
    ny, nx = data_2d.shape
    x, y = wcs.world_to_pixel_values(ra, dec)
    half = size // 2
    x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
    x1, y1 = x0 + size, y0 + size
    x0c, x1c = max(0, x0), min(nx, x1)
    y0c, y1c = max(0, y0), min(ny, y1)
    out = np.zeros((size, size), dtype=np.float32)
    if x1c > x0c and y1c > y0c:
        out[y0c - y0:y0c - y0 + (y1c - y0c), x0c - x0:x0c - x0 + (x1c - x0c)] = \
            data_2d[y0c:y1c, x0c:x1c]
    return out


def render_stci_tile(vis_data, y_data, j_data, wcs, sources, out_dir):
    fmt = OUTPUT_FORMAT.lower()
    n_ok = n_fail = 0
    for src in sources:
        tile_id = str(src['source_id']).split("_", 1)[0]
        tile_out = os.path.join(out_dir, tile_id)
        os.makedirs(tile_out, exist_ok=True)
        out_path = os.path.join(tile_out, f"{src['source_id']}.{fmt}")
        if os.path.exists(out_path):
            continue
        try:
            vis = _extract_cutout(vis_data, wcs, src["ra"], src["dec"], CUTOUT_PIXELS)
            y_cut = _extract_cutout(y_data, wcs, src["ra"], src["dec"], CUTOUT_PIXELS)
            j_cut = _extract_cutout(j_data, wcs, src["ra"], src["dec"], CUTOUT_PIXELS)

            red, green, blue, _ = normalize_raw_channels_common(j_cut, y_cut, vis)
            outputs = compose_pipeline(red, green, blue, config=STCI_CONFIG)
            final = outputs["13_final.tif"]

            final_uint8 = np.round(np.clip(np.flipud(final), 0.0, 1.0) * 255).astype(np.uint8)
            save_kw = {"quality": JPEG_QUALITY} if fmt == "jpg" else {}
            Image.fromarray(final_uint8).save(out_path, **save_kw)
            n_ok += 1
        except Exception:
            logging.exception("  STCI render failed for %s", src["source_id"])
            n_fail += 1

    if n_fail:
        logging.warning("  %d renders failed in %s", n_fail, out_dir)
    return n_ok


def process_tile(args):
    tile_id, sources, stci_out = args

    # Skip if all outputs exist (per-tile subfolder layout)
    fmt = OUTPUT_FORMAT.lower()
    tile_out_dir = os.path.join(stci_out, str(tile_id))
    existing = sum(1 for s in sources if os.path.exists(os.path.join(tile_out_dir, f"{s['source_id']}.{fmt}")))
    if existing == len(sources):
        return tile_id, 0, 0, 0

    ra0, dec0 = sources[0]["ra"], sources[0]["dec"]
    paths_3 = find_fits_paths_any_release(tile_id, BANDS_3, RELEASE_DIRS, BAND_TO_INST, ra=ra0, dec=dec0)
    if paths_3 is None:
        logging.warning("Tile %s: FITS not found — skipping %d sources", tile_id, len(sources))
        return tile_id, 0, 0, len(sources)

    # paths_3 is already [VIS, NIR_Y, NIR_J] from find_fits_paths_any_release
    handles = [fits.open(p, memmap=True) for p in paths_3]
    try:
        vis_data = handles[0][0].data.astype(np.float32)
        y_data = handles[1][0].data.astype(np.float32)
        j_data = handles[2][0].data.astype(np.float32)
        wcs = WCS(handles[0][0].header)
    finally:
        for h in handles:
            h.close()

    n_ok = render_stci_tile(vis_data, y_data, j_data, wcs, sources, stci_out)
    return tile_id, n_ok, 0, 0


def _init_worker():
    pass


def main():
    logging.info("Loading sources from %s", INPUT_PARQUET)
    df = load_sources(INPUT_PARQUET, COORDS_CSV)
    logging.info("%d sources across %d tiles", len(df), df["tile_id"].nunique())

    os.makedirs(STCI_OUT, exist_ok=True)

    work_items = [
        (int(tile_id),
         group[["source_id", "ra", "dec"]].to_dict("records"),
         STCI_OUT)
        for tile_id, group in df.groupby("tile_id")
    ]

    n_tiles = len(work_items)
    total_ok = total_skip = done = 0

    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for tile_id, n_ok, _, n_skip in pool.imap_unordered(process_tile, work_items):
            total_ok   += n_ok
            total_skip += n_skip
            done       += 1
            if done % 50 == 0 or done == n_tiles:
                logging.info(
                    "[%d/%d tiles] stci=%d skipped=%d",
                    done, n_tiles, total_ok, total_skip,
                )

    logging.info("Done. STCI images: %d  skipped: %d\n  → %s/", total_ok, total_skip, STCI_OUT)


if __name__ == "__main__":
    main()
