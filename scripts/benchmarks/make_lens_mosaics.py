"""
Build a per-pipeline mosaic (grid of labelled cutouts) for a sample of known
strong-lens candidates, comparing the four RGB pipelines benchmarked in
benchmark_rgb_pipelines.py:

    cutana_png_asinh, azulero_jpg, bulk_euclid_mtf, eummy_png

Lens catalogue: /media/user/Catalogs/all_cand_in_q1_randomtag_Mikerecentered_grade_A.csv

Run:
    python make_lens_mosaics.py [N_LENSES]
"""

import json
import logging
import os
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.insert(0, os.path.dirname(__file__))
import benchmark_rgb_pipelines as b
import make_azulero_cutouts as az

# ── CONFIG ────────────────────────────────────────────────────────────────────
LENS_CATALOGUE = "/media/user/Catalogs/all_cand_in_q1_randomtag_Mikerecentered_grade_A.csv"
N_LENSES        = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUTPUT_DIR      = "/media/user/cutana_dr1_pipeline/results/lens_mosaics_r2"
THUMB           = 150   # mosaic thumbnail size (px)
GRID_COLS       = 5
# ── END CONFIG ────────────────────────────────────────────────────────────────


def build_lens_sample(n: int) -> pd.DataFrame:
    df = pd.read_csv(LENS_CATALOGUE)
    df = df.head(n).copy()
    df = df.rename(columns={"DEC": "Dec"})
    df["diameter_pixel"] = 101

    fits_paths_col, source_ids = [], []
    for _, row in df.iterrows():
        tile_index = int(row["tile_index"])
        paths = az.find_fits_paths_any_release(tile_index, b.BANDS, ra=row["RA"], dec=row["Dec"])
        fits_paths_col.append(json.dumps(paths) if paths is not None else None)
        source_ids.append(f"{tile_index}_{str(row['object_id']).replace('-', 'NEG')}")

    df["fits_file_paths"] = fits_paths_col
    df["SourceID"] = source_ids
    df["tile_index"] = df["tile_index"].astype(int)

    before = len(df)
    df = df.dropna(subset=["fits_file_paths"]).reset_index(drop=True)
    logging.info(f"{before - len(df)} lenses dropped (missing IYJH files). {len(df)} remain.")
    return df


def build_mosaic(image_paths: list, labels: list, out_path: str, title: str) -> None:
    n = len(image_paths)
    cols = GRID_COLS
    rows = (n + cols - 1) // cols
    label_h = 18
    cell_w, cell_h = THUMB, THUMB + label_h

    mosaic = Image.new("RGB", (cols * cell_w, rows * cell_h + 30), "black")
    draw = ImageDraw.Draw(mosaic)
    draw.text((10, 5), title, fill="white")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, 30 + r * cell_h
        try:
            im = Image.open(path).convert("RGB").resize((THUMB, THUMB))
            mosaic.paste(im, (x0, y0))
        except Exception:
            logging.exception(f"Could not open {path}")
        draw.text((x0 + 4, y0 + THUMB + 2), label, fill="white", font=font)

    mosaic.save(out_path)
    logging.info(f"Saved mosaic: {out_path} ({n} cutouts, {cols}x{rows} grid)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample = build_lens_sample(N_LENSES)
    sample.to_csv(os.path.join(OUTPUT_DIR, "lens_sample.csv"), index=False)
    logging.info(f"Lens sample: {len(sample)} sources across {sample['tile_index'].nunique()} tiles")

    labels = [row["IAU_NAME"].replace("EUCL J", "") if "IAU_NAME" in sample.columns and pd.notna(row["IAU_NAME"])
              else row["SourceID"] for _, row in sample.iterrows()]

    b.OUTPUT_DIR = OUTPUT_DIR

    # 1) Cutana PNG
    logging.info("=== Cutana PNG (asinh) ===")
    b.benchmark_cutana_png(sample)
    paths = [os.path.join(OUTPUT_DIR, "cutana_png", f"{sid}.png") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_cutana_png_asinh.png"), "Cutana PNG (asinh)")

    # 2) Azulero JPEG
    logging.info("=== Azulero JPEG ===")
    b.benchmark_azulero_jpg(sample)
    paths = [os.path.join(OUTPUT_DIR, "azulero_jpg", "jpgs", f"{sid}.jpg") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_azulero_jpg.png"), "Azulero JPEG")

    # 3) bulk-euclid MTF
    logging.info("=== bulk-euclid MTF ===")
    b.benchmark_bulk_euclid_mtf(sample)
    paths = [os.path.join(OUTPUT_DIR, "bulk_euclid_mtf", f"{sid}.jpg") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_bulk_euclid_mtf.png"), "bulk-euclid MTF (sw_mtf_vis_y_j)")

    # 4) eummy PNG
    logging.info("=== eummy PNG (this is slow: ~40-50s/tile) ===")
    b.benchmark_eummy_png(sample)
    paths = []
    for _, row in sample.iterrows():
        paths.append(os.path.join(OUTPUT_DIR, "eummy_png", f"tile_{row['tile_index']}", f"{row['SourceID']}.png"))
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_eummy_png.png"), "eummy PNG")

    logging.info(f"Done. Mosaics written to {OUTPUT_DIR}/mosaic_*.png")


if __name__ == "__main__":
    main()
