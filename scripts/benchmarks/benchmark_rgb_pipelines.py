"""
Benchmark RGB image-generation pipelines on a small sample of sources from
catalogue_200k.csv: how fast can each pipeline produce an RGB cutout at
VIS resolution/quality, with colour information from the NIR bands, using
an MTF-like (asinh) stretch?

Pipelines compared:
  A) Cutana RGB PNG composite (vectorized, batch, asinh normalisation)
  B) Azulero JPEG (per-tile in-memory colour pipeline: inpaint, sharpen,
     asinh stretch, IYJH->LRGB->RGB blend)
  C) bulk-euclid-cutouts MTF JPEG ("sw_mtf_vis_y_j": per-band MTF stretch,
     RGB=(J,Y,VIS), VIS used as the LAB luminance channel -- VIS quality,
     colour from Y/J)

Run:
    python benchmark_rgb_pipelines.py [N_SOURCES]
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import make_azulero_cutouts as az
import make_eummy_cutouts as eum
from jobs_200k.make_azulero_jpg_200k import process_tile as azulero_process_tile

sys.path.insert(0, "/media/user/bulk-euclid-cutouts")
from bulk_euclid.utils import cutout_utils as bec
from bulk_euclid.utils import morphology_utils_ou_mer as m_utils

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CATALOGUE = "/media/user/cutana_dr1_pipeline/results/200k/cutana_output_200k/catalogue_200k.csv"
OUTPUT_DIR      = "/media/user/cutana_dr1_pipeline/results/benchmarks"
N_SOURCES       = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED            = 42
CUTOUT_PIXELS   = 101  # native VIS resolution (no resampling)

BANDS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
# RGB mapping for the Cutana composite (mirrors azulero's I->B, Y->G, J->R)
CHANNEL_WEIGHTS = {
    "VIS":   [0.0, 0.0, 1.0],
    "NIR-Y": [0.0, 1.0, 0.0],
    "NIR-J": [1.0, 0.0, 0.0],
    "NIR-H": [0.0, 0.0, 0.0],
}
# ── END CONFIG ────────────────────────────────────────────────────────────────


def build_sample(df: pd.DataFrame, n: int = N_SOURCES, seed: int = SEED, exclude_tiles: set = frozenset()) -> pd.DataFrame:
    df = df.copy()
    df["tile_index"] = df["SourceID"].str.split("_").str[0].astype(int)
    pool = df[~df["tile_index"].isin(exclude_tiles)]
    sample = pool.sample(n=n, random_state=seed).copy()

    fits_paths_col = []
    for tile_index in sample["tile_index"]:
        paths = az.find_fits_paths_any_release(int(tile_index), BANDS)
        fits_paths_col.append(json.dumps(paths) if paths is not None else None)
    sample["fits_file_paths"] = fits_paths_col

    before = len(sample)
    sample = sample.dropna(subset=["fits_file_paths"]).reset_index(drop=True)
    logging.info(f"{before - len(sample)} sources dropped (missing IYJH files). {len(sample)} remain.")
    return sample


def build_disjoint_samples(df: pd.DataFrame, n: int = N_SOURCES, seed: int = SEED, n_samples: int = 3) -> list[pd.DataFrame]:
    """Build n_samples samples of n sources each, drawn so that no two samples
    share a tile_index -- i.e. each pipeline gets its own set of never-yet-touched
    tile mosaics, so the first ("cold") run of each pipeline is comparable."""
    samples = []
    used_tiles = set()
    for i in range(n_samples):
        sample = build_sample(df, n=n, seed=seed + i, exclude_tiles=used_tiles)
        used_tiles |= set(sample["tile_index"])
        samples.append(sample)
    return samples


def benchmark_cutana_png(sample: pd.DataFrame) -> float:
    from cutana import create_cutouts_direct, get_default_config

    out_dir = os.path.join(OUTPUT_DIR, "cutana_png")
    os.makedirs(out_dir, exist_ok=True)

    catalogue = sample[["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"]]

    config = get_default_config()
    config.target_resolution = CUTOUT_PIXELS
    config.selected_extensions = [{"name": b.replace("_", "-"), "ext": "PrimaryHDU"} for b in BANDS]
    config.channel_weights = {b.replace("_", "-"): CHANNEL_WEIGHTS[b.replace("_", "-")] for b in BANDS}
    config.normalisation_method = "asinh"

    t0 = time.time()
    results = create_cutouts_direct(catalogue, config)

    n_written = 0
    for batch_result in results:
        cutouts = batch_result["cutouts"]
        metadata = batch_result["metadata"]
        for cutout, meta in zip(cutouts, metadata):
            rgb_uint8 = (np.clip(cutout, 0.0, 1.0) * 255).round().astype(np.uint8)
            Image.fromarray(rgb_uint8).save(os.path.join(out_dir, f"{meta['source_id']}.png"))
            n_written += 1
    elapsed = time.time() - t0

    logging.info(f"[Cutana PNG] {n_written}/{len(sample)} written in {elapsed:.2f}s "
                  f"({elapsed/len(sample)*1000:.1f} ms/image)")
    return elapsed


def benchmark_azulero_jpg(sample: pd.DataFrame) -> float:
    out_dir = os.path.join(OUTPUT_DIR, "azulero_jpg")
    os.makedirs(out_dir, exist_ok=True)

    # Patch the OUTPUT_DIR used by process_tile (it builds Path(OUTPUT_DIR)/"jpgs")
    import jobs_200k.make_azulero_jpg_200k as azjpg
    azjpg.OUTPUT_DIR = out_dir
    azjpg.CUTOUT_PIXELS = CUTOUT_PIXELS
    os.makedirs(os.path.join(out_dir, "jpgs"), exist_ok=True)

    work_items = []
    for tile_index, group in sample.groupby("tile_index"):
        ra, dec = group["RA"].iloc[0], group["Dec"].iloc[0]
        fits_paths = az.find_fits_paths_any_release(int(tile_index), az.BANDS, ra=ra, dec=dec)
        sources = list(zip(group["SourceID"], group["RA"], group["Dec"]))
        work_items.append((int(tile_index), fits_paths, sources))

    t0 = time.time()
    total_ok, total_fail = 0, 0
    for item in work_items:
        _, n_ok, n_fail = azulero_process_tile(item)
        total_ok += n_ok
        total_fail += n_fail
    elapsed = time.time() - t0

    logging.info(f"[Azulero JPEG] {total_ok}/{len(sample)} written ({total_fail} failed/no-NIR-coverage) "
                  f"in {elapsed:.2f}s ({elapsed/len(sample)*1000:.1f} ms/image)")
    return elapsed


def _process_tile_bulk_euclid(args):
    tile_index, fits_paths, sources, out_dir = args
    half = CUTOUT_PIXELS // 2

    # fits_paths order is [VIS, NIR_Y, NIR_J, NIR_H] -- only need VIS, Y, J
    handles = [fits.open(p, memmap=True) for p in fits_paths[:3]]
    data = [h[0].data for h in handles]
    wcs = WCS(handles[0][0].header)
    ny, nx = data[0].shape

    n_ok, n_fail = 0, 0
    for source_id, ra, dec in sources:
        x, y = wcs.world_to_pixel_values(ra, dec)
        x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
        x1, y1 = x0 + CUTOUT_PIXELS, y0 + CUTOUT_PIXELS

        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            n_fail += 1
            continue

        try:
            vis_im, y_im, j_im = (d[y0:y1, x0:x1].astype(np.float32) for d in data)

            vis_mtf = bec.apply_MTF(vis_im)
            y_mtf = bec.apply_MTF(y_im)
            j_mtf = bec.apply_MTF(j_im)

            rgb_mtf = np.stack([j_mtf, y_mtf, vis_mtf], axis=2).astype(np.uint8)
            lab_mtf = bec.replace_luminosity_channel(rgb_mtf, rgb_channel_for_luminosity=2, desaturate_speckles=False)

            Image.fromarray(lab_mtf).save(os.path.join(out_dir, f"{source_id}.jpg"), quality=90)
            n_ok += 1
        except Exception:
            logging.exception(f"Failed: {source_id}")
            n_fail += 1

    for h in handles:
        h.close()

    return tile_index, n_ok, n_fail


def benchmark_bulk_euclid_mtf(sample: pd.DataFrame) -> float:
    out_dir = os.path.join(OUTPUT_DIR, "bulk_euclid_mtf")
    os.makedirs(out_dir, exist_ok=True)

    work_items = []
    for tile_index, group in sample.groupby("tile_index"):
        ra, dec = group["RA"].iloc[0], group["Dec"].iloc[0]
        fits_paths = az.find_fits_paths_any_release(int(tile_index), az.BANDS, ra=ra, dec=dec)
        sources = list(zip(group["SourceID"], group["RA"], group["Dec"]))
        work_items.append((int(tile_index), fits_paths, sources, out_dir))

    t0 = time.time()
    total_ok, total_fail = 0, 0
    for item in work_items:
        _, n_ok, n_fail = _process_tile_bulk_euclid(item)
        total_ok += n_ok
        total_fail += n_fail
    elapsed = time.time() - t0

    logging.info(f"[bulk-euclid MTF] {total_ok}/{len(sample)} written ({total_fail} failed) "
                  f"in {elapsed:.2f}s ({elapsed/len(sample)*1000:.1f} ms/image)")
    return elapsed


def _process_tile_bulk_euclid_arcsinh_vis_y(args):
    tile_index, fits_paths, sources, out_dir = args
    half = CUTOUT_PIXELS // 2

    # fits_paths order is [VIS, NIR_Y, NIR_J, NIR_H] -- only need VIS, Y
    handles = [fits.open(p, memmap=True) for p in fits_paths[:2]]
    data = [h[0].data for h in handles]
    wcs = WCS(handles[0][0].header)
    ny, nx = data[0].shape

    n_ok, n_fail = 0, 0
    for source_id, ra, dec in sources:
        x, y = wcs.world_to_pixel_values(ra, dec)
        x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
        x1, y1 = x0 + CUTOUT_PIXELS, y0 + CUTOUT_PIXELS

        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            n_fail += 1
            continue

        try:
            vis_im, y_im = (d[y0:y1, x0:x1].astype(np.float32) for d in data)

            # "sw_arcsinh_vis_y": arcsinh dynamic-range compression (vis_q=500, nisp_q=1),
            # RGB = (Y, mean(VIS,Y), VIS), then VIS replaces the LAB luminance channel
            vis_y_rgb = bec.make_composite_cutout(vis_im, y_im, vis_q=500, nisp_q=1)
            vis_y_rgb_lab = bec.replace_luminosity_channel(vis_y_rgb, rgb_channel_for_luminosity=2, desaturate_speckles=False)

            Image.fromarray(vis_y_rgb_lab).save(os.path.join(out_dir, f"{source_id}.jpg"), quality=90)
            n_ok += 1
        except Exception:
            logging.exception(f"Failed: {source_id}")
            n_fail += 1

    for h in handles:
        h.close()

    return tile_index, n_ok, n_fail


def benchmark_bulk_euclid_arcsinh_vis_y(sample: pd.DataFrame) -> float:
    out_dir = os.path.join(OUTPUT_DIR, "bulk_euclid_arcsinh_vis_y")
    os.makedirs(out_dir, exist_ok=True)

    work_items = []
    for tile_index, group in sample.groupby("tile_index"):
        ra, dec = group["RA"].iloc[0], group["Dec"].iloc[0]
        fits_paths = az.find_fits_paths_any_release(int(tile_index), az.BANDS, ra=ra, dec=dec)
        sources = list(zip(group["SourceID"], group["RA"], group["Dec"]))
        work_items.append((int(tile_index), fits_paths, sources, out_dir))

    t0 = time.time()
    total_ok, total_fail = 0, 0
    for item in work_items:
        _, n_ok, n_fail = _process_tile_bulk_euclid_arcsinh_vis_y(item)
        total_ok += n_ok
        total_fail += n_fail
    elapsed = time.time() - t0

    logging.info(f"[bulk-euclid arcsinh VIS+Y] {total_ok}/{len(sample)} written ({total_fail} failed) "
                  f"in {elapsed:.2f}s ({elapsed/len(sample)*1000:.1f} ms/image)")
    return elapsed


def _process_tile_bulk_euclid_arcsinh_vis_only(args):
    tile_index, fits_paths, sources, out_dir = args
    half = CUTOUT_PIXELS // 2

    handles = [fits.open(fits_paths[0], memmap=True)]  # VIS only
    data = handles[0][0].data
    wcs = WCS(handles[0][0].header)
    ny, nx = data.shape

    n_ok, n_fail = 0, 0
    for source_id, ra, dec in sources:
        x, y = wcs.world_to_pixel_values(ra, dec)
        x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
        x1, y1 = x0 + CUTOUT_PIXELS, y0 + CUTOUT_PIXELS

        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            n_fail += 1
            continue

        try:
            vis_im = data[y0:y1, x0:x1].astype(np.float32)

            # "sw_arcsinh_vis_only": arcsinh dynamic-range compression (q=500), greyscale
            vis_uint8 = m_utils.make_vis_only_cutout(vis_im, q=500)

            Image.fromarray(vis_uint8, mode="L").save(os.path.join(out_dir, f"{source_id}.jpg"), quality=90)
            n_ok += 1
        except Exception:
            logging.exception(f"Failed: {source_id}")
            n_fail += 1

    for h in handles:
        h.close()

    return tile_index, n_ok, n_fail


def benchmark_bulk_euclid_arcsinh_vis_only(sample: pd.DataFrame) -> float:
    out_dir = os.path.join(OUTPUT_DIR, "bulk_euclid_arcsinh_vis_only")
    os.makedirs(out_dir, exist_ok=True)

    work_items = []
    for tile_index, group in sample.groupby("tile_index"):
        ra, dec = group["RA"].iloc[0], group["Dec"].iloc[0]
        fits_paths = az.find_fits_paths_any_release(int(tile_index), az.BANDS, ra=ra, dec=dec)
        sources = list(zip(group["SourceID"], group["RA"], group["Dec"]))
        work_items.append((int(tile_index), fits_paths, sources, out_dir))

    t0 = time.time()
    total_ok, total_fail = 0, 0
    for item in work_items:
        _, n_ok, n_fail = _process_tile_bulk_euclid_arcsinh_vis_only(item)
        total_ok += n_ok
        total_fail += n_fail
    elapsed = time.time() - t0

    logging.info(f"[bulk-euclid arcsinh VIS-only] {total_ok}/{len(sample)} written ({total_fail} failed) "
                  f"in {elapsed:.2f}s ({elapsed/len(sample)*1000:.1f} ms/image)")
    return elapsed


def benchmark_eummy_png(sample: pd.DataFrame) -> float:
    """eummy (https://pypi.org/project/eummy/): a CLI tool that loads a full
    MER tile (all 4 IYJH bands) and produces an arcsinh-stretched colour PNG
    per cutout. Invoked once per tile via subprocess, on a workspace of
    symlinked FITS files + a small RA/Dec catalogue (mirrors make_eummy_cutouts.py)."""
    out_dir = os.path.join(OUTPUT_DIR, "eummy_png")
    os.makedirs(out_dir, exist_ok=True)

    eum.RA_COL = "RA"
    eum.DEC_COL = "Dec"
    eum.CUTOUT_ARCSEC = 10.0  # ~101 px at the VIS pixel scale (0.1"/px)

    t0 = time.time()
    total_ok, total_fail = 0, 0
    for tile_index, group in sample.groupby("tile_index"):
        ra, dec = group["RA"].iloc[0], group["Dec"].iloc[0]
        fits_paths = az.find_fits_paths_any_release(int(tile_index), BANDS, ra=ra, dec=dec)
        tile_dir = os.path.join(out_dir, f"tile_{tile_index}")
        eum.setup_tile_workspace(int(tile_index), fits_paths, tile_dir)

        catalog_path = os.path.join(tile_dir, "cutout_catalog.fits")
        eum.write_tile_catalog(group, catalog_path)

        try:
            eum.run_eummy_for_tile(tile_dir, catalog_path)
            n_renamed = eum.rename_cutouts(tile_dir, group)
            total_ok += n_renamed
            total_fail += len(group) - n_renamed
        except Exception:
            logging.exception(f"eummy failed for tile {tile_index}")
            total_fail += len(group)
    elapsed = time.time() - t0

    logging.info(f"[eummy PNG] {total_ok}/{len(sample)} written ({total_fail} failed) "
                  f"in {elapsed:.2f}s ({elapsed/len(sample)*1000:.1f} ms/image)")
    return elapsed


PIPELINES = {
    "cutana_png_asinh":           benchmark_cutana_png,
    "azulero_jpg":                benchmark_azulero_jpg,
    "bulk_euclid_mtf":            benchmark_bulk_euclid_mtf,
    "bulk_euclid_arcsinh_vis_y":  benchmark_bulk_euclid_arcsinh_vis_y,
    "bulk_euclid_arcsinh_vis_only": benchmark_bulk_euclid_arcsinh_vis_only,
    "eummy_png":                  benchmark_eummy_png,
}


def main():
    logging.info(f"Loading catalogue from {INPUT_CATALOGUE}")
    df = pd.read_csv(INPUT_CATALOGUE)

    logging.info(f"Building 3 disjoint {N_SOURCES}-source samples (seed={SEED}) "
                  "-- one per pipeline, no shared tiles -> fair cold-cache comparison")
    samples = build_disjoint_samples(df, n=N_SOURCES, seed=SEED, n_samples=len(PIPELINES))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, sample in zip(PIPELINES, samples):
        sample.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_sample_{name}.csv"), index=False)

    rows = []
    for (name, fn), sample in zip(PIPELINES.items(), samples):
        n = len(sample)
        cold_time = fn(sample)  # first touch of this pipeline's tiles -> cold
        warm_time = fn(sample)  # same tiles again -> warm (compute-only)
        rows.append({
            "pipeline": name, "n_images": n,
            "cold_total_s": cold_time, "cold_ms_per_image": cold_time / n * 1000,
            "warm_total_s": warm_time, "warm_ms_per_image": warm_time / n * 1000,
        })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False))
    logging.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
