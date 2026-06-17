"""
Generate azulero colour-composite JPEGs for the 200k-object sample, processing
each tile's 4 IYJH mosaics once (in memory, via memmap) and slicing per-source
cutouts directly -- avoids per-source file I/O / symlink workspaces.

Run:
    python make_azulero_jpg_200k.py
"""

import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import make_azulero_cutouts as az
from azulero.image import color, io as az_io, mask as az_mask
from azulero.tools import parsing as az_parsing

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CATALOGUE = "/media/user/cutana_dr1_pipeline/results/200k/cutana_output_200k/catalogue_200k.csv"
OUTPUT_DIR      = "/media/user/cutana_dr1_pipeline/results/200k/azulero_output_200k"
CUTOUT_PIXELS   = 101  # matches diameter_pixel in the input catalogue
N_WORKERS       = 6
# ── END CONFIG ────────────────────────────────────────────────────────────────


def build_transform():
    return color.Transform(
        iyjh_zero_points=np.array(az.AZUL_ZERO),
        iyjh_scaling=np.array(az.AZUL_SCALING),
        iyjh_fwhm=np.array(az.AZUL_FWHM),
        sharpen_strength=az.AZUL_SHARPEN,
        nir_to_l=az.AZUL_NIRL,
        i_to_b=az.AZUL_IB,
        y_to_g=az.AZUL_YG,
        j_to_r=az.AZUL_JR,
        hue=az.AZUL_HUE,
        saturation=az.AZUL_SATURATION,
        stretch=az.AZUL_STRETCH,
        bw=np.array([az.AZUL_OFFSET, az.AZUL_WHITE]),
        curves=[],
    )


def render_jpg(iyjh: np.ndarray, transform, out_path: Path) -> None:
    dead = az_mask.dead_pixels(iyjh)
    iyjh[0] = az_mask.inpaint(iyjh[0], dead[0])
    nir_dead = dead[1] | dead[2] | dead[3]
    iyjh[1:] = az_mask.inpaint(iyjh[1:], nir_dead, 0)

    iyjh = color.sharpen(iyjh, transform.iyjh_fwhm / 2.355, transform.sharpen_strength)
    iyjh = color.stretch_iyjh(iyjh, transform)

    lbgr = color.iyjh_to_lbgr(iyjh, transform)
    bgr = color.lbgr_to_bgr(lbgr, transform)
    az_mask.resaturate(bgr[dead[0]])

    rgb = bgr[..., ::-1]
    for i in range(len(az.AZUL_CURVES)):
        knots = az_parsing.parse_map(az.AZUL_CURVES[i])
        knots.insert(0, [0, 0])
        knots.append([1, 1])
        rgb[:, :, i] = color.adjust_curve(rgb[:, :, i], knots)

    az_io.write_rgb(rgb, out_path)


def process_tile(args):
    tile_index, fits_paths, sources = args
    transform = build_transform()
    jpg_dir = Path(OUTPUT_DIR) / "jpgs"

    # Open all 4 channel mosaics as memmaps (cheap; pages loaded on demand)
    handles = [fits.open(p, memmap=True) for p in fits_paths]
    data = [h[0].data for h in handles]
    wcs = WCS(handles[0][0].header)
    ny, nx = data[0].shape

    half = CUTOUT_PIXELS // 2
    n_ok, n_fail = 0, 0
    for source_id, ra, dec in sources:
        x, y = wcs.world_to_pixel_values(ra, dec)
        x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
        x1, y1 = x0 + CUTOUT_PIXELS, y0 + CUTOUT_PIXELS

        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            n_fail += 1
            continue

        try:
            iyjh = np.stack(
                [d[y0:y1, x0:x1].astype(np.float32) for d in data]
            )
            render_jpg(iyjh, transform, jpg_dir / f"{source_id}.jpg")
            n_ok += 1
        except Exception:
            logging.exception(f"Failed: {source_id}")
            n_fail += 1

    for h in handles:
        h.close()

    return tile_index, n_ok, n_fail


def main():
    logging.info(f"Loading catalogue from {INPUT_CATALOGUE}")
    df = pd.read_csv(INPUT_CATALOGUE)
    df["tile_index"] = df["SourceID"].str.split("_").str[0].astype(int)
    logging.info(f"Loaded {len(df)} sources across {df['tile_index'].nunique()} tiles")

    jpg_dir = os.path.join(OUTPUT_DIR, "jpgs")
    os.makedirs(jpg_dir, exist_ok=True)

    # Build per-tile work items: (tile_index, [VIS,NIR-Y,NIR-J,NIR-H] paths, [(SourceID, RA, Dec), ...])
    work_items = []
    skipped_tiles = 0
    for tile_index, group in df.groupby("tile_index"):
        fits_paths = az.find_fits_paths_any_release(int(tile_index), az.BANDS)
        if fits_paths is None:
            skipped_tiles += 1
            continue
        sources = list(zip(group["SourceID"], group["RA"], group["Dec"]))
        work_items.append((int(tile_index), fits_paths, sources))

    logging.info(f"{len(work_items)} tiles to process ({skipped_tiles} skipped, missing IYJH)")

    total_ok, total_fail, n_tiles_done = 0, 0, 0
    with mp.Pool(N_WORKERS, initializer=lambda: __import__("cv2").setNumThreads(1)) as pool:
        for tile_index, n_ok, n_fail in pool.imap_unordered(process_tile, work_items):
            total_ok += n_ok
            total_fail += n_fail
            n_tiles_done += 1
            if n_tiles_done % 10 == 0 or n_tiles_done == len(work_items):
                logging.info(
                    f"[{n_tiles_done}/{len(work_items)} tiles] "
                    f"ok={total_ok} fail={total_fail}"
                )

    logging.info(f"Done. {total_ok} JPEGs written to {jpg_dir}, {total_fail} sources failed/skipped")


if __name__ == "__main__":
    main()
