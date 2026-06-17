"""
Same as make_lens_mosaics.py + make_combined_lens_mosaic.py, but forcing all
cutouts to be taken from euclid_q1/Q1_R1 (instead of the default R2-first lookup), for
direct comparison against the R2-based mosaics in results/lens_mosaics_r2/.

Uses the same 50-lens sample (lens_sample.csv) so rows line up 1:1 with the
R2 combined mosaic.

Run:
    python make_lens_mosaics_q1r1.py
"""

import json
import logging
import os
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import benchmark_rgb_pipelines as b
import make_azulero_cutouts as az

# ── CONFIG ────────────────────────────────────────────────────────────────────
R2_OUTPUT_DIR = "/media/user/cutana_dr1_pipeline/results/lens_mosaics_r2"
OUTPUT_DIR    = "/media/user/cutana_dr1_pipeline/results/lens_mosaics_q1r1"
R1_RELEASE_DIR = "/media/home/data/euclid_q1/Q1_R1"
THUMB         = 150
GRID_COLS     = 5

THUMB_C   = 120
LABEL_W   = 175
HEADER_H  = 30

PIPELINES_GRID = [
    ("cutana_png_asinh",            "Cutana\n(asinh)",        lambda sid, tile: os.path.join(OUTPUT_DIR, "cutana_png", f"{sid}.png")),
    ("azulero_jpg",                 "Azulero",                lambda sid, tile: os.path.join(OUTPUT_DIR, "azulero_jpg", "jpgs", f"{sid}.jpg")),
    ("bulk_euclid_mtf",             "bulk-euclid\nMTF",       lambda sid, tile: os.path.join(OUTPUT_DIR, "bulk_euclid_mtf", f"{sid}.jpg")),
    ("bulk_euclid_arcsinh_vis_y",   "bulk-euclid\narcsinh VIS+Y", lambda sid, tile: os.path.join(OUTPUT_DIR, "bulk_euclid_arcsinh_vis_y", f"{sid}.jpg")),
    ("bulk_euclid_arcsinh_vis_only","bulk-euclid\narcsinh VIS-only", lambda sid, tile: os.path.join(OUTPUT_DIR, "bulk_euclid_arcsinh_vis_only", f"{sid}.jpg")),
    ("eummy_png",                   "eummy",                  lambda sid, tile: os.path.join(OUTPUT_DIR, "eummy_png", f"tile_{tile}", f"{sid}.png")),
]
# ── END CONFIG ────────────────────────────────────────────────────────────────


def build_lens_sample_q1r1() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(R2_OUTPUT_DIR, "lens_sample.csv"))

    fits_paths_col = []
    for _, row in df.iterrows():
        tile_index = int(row["tile_index"])
        paths = az.find_fits_paths(tile_index, b.BANDS, R1_RELEASE_DIR, ra=row["RA"], dec=row["Dec"])
        fits_paths_col.append(json.dumps(paths) if paths is not None else None)
    df["fits_file_paths"] = fits_paths_col

    before = len(df)
    df = df.dropna(subset=["fits_file_paths"]).reset_index(drop=True)
    logging.info(f"{before - len(df)} lenses dropped (missing R1 IYJH files). {len(df)} remain.")
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


def build_combined_mosaic(sample: pd.DataFrame) -> None:
    n = len(sample)
    n_cols = len(PIPELINES_GRID)

    width = LABEL_W + n_cols * THUMB_C
    height = HEADER_H + n * THUMB_C

    mosaic = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(mosaic)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for c, (_, header, _) in enumerate(PIPELINES_GRID):
        x0 = LABEL_W + c * THUMB_C
        for li, line in enumerate(header.split("\n")):
            draw.text((x0 + 4, 2 + li * 10), line, fill="white", font=font)

    for r, (_, row) in enumerate(sample.iterrows()):
        sid = row["SourceID"]
        tile = int(row["tile_index"])
        label = row["IAU_NAME"].replace("EUCL J", "") if pd.notna(row.get("IAU_NAME")) else sid
        label = f"[{r}] {label}"

        y0 = HEADER_H + r * THUMB_C
        draw.text((4, y0 + THUMB_C // 2 - 6), label, fill="white", font=font)

        for c, (_, _, path_fn) in enumerate(PIPELINES_GRID):
            x0 = LABEL_W + c * THUMB_C
            path = path_fn(sid, tile)
            try:
                im = Image.open(path).convert("RGB").resize((THUMB_C, THUMB_C))
                mosaic.paste(im, (x0, y0))
            except Exception:
                pass

    out_path = os.path.join(OUTPUT_DIR, "mosaic_combined_all_pipelines_q1r1.png")
    mosaic.save(out_path)
    logging.info(f"Saved combined R1 mosaic: {out_path} ({n} lenses x {n_cols} pipelines, {width}x{height}px)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample = build_lens_sample_q1r1()
    sample.to_csv(os.path.join(OUTPUT_DIR, "lens_sample_q1r1.csv"), index=False)
    logging.info(f"R1 lens sample: {len(sample)} sources across {sample['tile_index'].nunique()} tiles")

    labels = [row["IAU_NAME"].replace("EUCL J", "") if "IAU_NAME" in sample.columns and pd.notna(row["IAU_NAME"])
              else row["SourceID"] for _, row in sample.iterrows()]

    b.OUTPUT_DIR = OUTPUT_DIR

    logging.info("=== Cutana PNG (asinh) ===")
    b.benchmark_cutana_png(sample)
    paths = [os.path.join(OUTPUT_DIR, "cutana_png", f"{sid}.png") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_cutana_png_asinh_q1r1.png"), "Cutana PNG (asinh) - Q1_R1")

    logging.info("=== Azulero JPEG ===")
    b.benchmark_azulero_jpg(sample)
    paths = [os.path.join(OUTPUT_DIR, "azulero_jpg", "jpgs", f"{sid}.jpg") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_azulero_jpg_q1r1.png"), "Azulero JPEG - Q1_R1")

    logging.info("=== bulk-euclid MTF ===")
    b.benchmark_bulk_euclid_mtf(sample)
    paths = [os.path.join(OUTPUT_DIR, "bulk_euclid_mtf", f"{sid}.jpg") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_bulk_euclid_mtf_q1r1.png"), "bulk-euclid MTF - Q1_R1")

    logging.info("=== bulk-euclid arcsinh VIS+Y ===")
    b.benchmark_bulk_euclid_arcsinh_vis_y(sample)
    paths = [os.path.join(OUTPUT_DIR, "bulk_euclid_arcsinh_vis_y", f"{sid}.jpg") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_bulk_euclid_arcsinh_vis_y_q1r1.png"), "bulk-euclid arcsinh VIS+Y - Q1_R1")

    logging.info("=== bulk-euclid arcsinh VIS-only ===")
    b.benchmark_bulk_euclid_arcsinh_vis_only(sample)
    paths = [os.path.join(OUTPUT_DIR, "bulk_euclid_arcsinh_vis_only", f"{sid}.jpg") for sid in sample["SourceID"]]
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_bulk_euclid_arcsinh_vis_only_q1r1.png"), "bulk-euclid arcsinh VIS-only - Q1_R1")

    logging.info("=== eummy PNG (this is slow: ~40-50s/tile) ===")
    b.benchmark_eummy_png(sample)
    paths = []
    for _, row in sample.iterrows():
        paths.append(os.path.join(OUTPUT_DIR, "eummy_png", f"tile_{row['tile_index']}", f"{row['SourceID']}.png"))
    build_mosaic(paths, labels, os.path.join(OUTPUT_DIR, "mosaic_eummy_png_q1r1.png"), "eummy PNG - Q1_R1")

    build_combined_mosaic(sample)

    logging.info(f"Done. Mosaics written to {OUTPUT_DIR}/mosaic_*.png")


if __name__ == "__main__":
    main()
