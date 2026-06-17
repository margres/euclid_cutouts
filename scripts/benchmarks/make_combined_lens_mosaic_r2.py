"""
Build the indexed combined mosaic (one lens per row, one pipeline per column)
for R2, reusing the cutouts already written by make_lens_mosaics.py under
results/lens_mosaics_r2/. This mirrors mosaic_combined_all_pipelines_r1.png /
mosaic_combined_all_pipelines_q1r1.png for direct release-to-release comparison.

Run:
    python make_combined_lens_mosaic_r2.py
"""

import logging
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

R2_DIR     = "/media/user/cutana_dr1_pipeline/results/lens_mosaics_r2"
OUTPUT_DIR = "/media/user/cutana_dr1_pipeline/results/lens_mosaics_r2"
THUMB      = 120
LABEL_W    = 175
HEADER_H   = 30

PIPELINES = [
    ("cutana_png_asinh",            "Cutana\n(asinh)",        lambda sid, tile: os.path.join(R2_DIR, "cutana_png", f"{sid}.png")),
    ("azulero_jpg",                 "Azulero",                lambda sid, tile: os.path.join(R2_DIR, "azulero_jpg", "jpgs", f"{sid}.jpg")),
    ("bulk_euclid_mtf",             "bulk-euclid\nMTF",       lambda sid, tile: os.path.join(R2_DIR, "bulk_euclid_mtf", f"{sid}.jpg")),
    ("bulk_euclid_arcsinh_vis_y",   "bulk-euclid\narcsinh VIS+Y", lambda sid, tile: os.path.join(R2_DIR, "bulk_euclid_arcsinh_vis_y", f"{sid}.jpg")),
    ("bulk_euclid_arcsinh_vis_only","bulk-euclid\narcsinh VIS-only", lambda sid, tile: os.path.join(R2_DIR, "bulk_euclid_arcsinh_vis_only", f"{sid}.jpg")),
    ("eummy_png",                   "eummy",                  lambda sid, tile: os.path.join(R2_DIR, "eummy_png", f"tile_{tile}", f"{sid}.png")),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample = pd.read_csv(os.path.join(R2_DIR, "lens_sample.csv"))
    n = len(sample)
    n_cols = len(PIPELINES)

    width = LABEL_W + n_cols * THUMB
    height = HEADER_H + n * THUMB

    mosaic = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(mosaic)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for c, (_, header, _) in enumerate(PIPELINES):
        x0 = LABEL_W + c * THUMB
        for li, line in enumerate(header.split("\n")):
            draw.text((x0 + 4, 2 + li * 10), line, fill="white", font=font)

    for r, (_, row) in enumerate(sample.iterrows()):
        sid = row["SourceID"]
        tile = int(row["tile_index"])
        label = row["IAU_NAME"].replace("EUCL J", "") if pd.notna(row.get("IAU_NAME")) else sid
        label = f"[{r}] {label}"

        y0 = HEADER_H + r * THUMB
        draw.text((4, y0 + THUMB // 2 - 6), label, fill="white", font=font)

        for c, (_, _, path_fn) in enumerate(PIPELINES):
            x0 = LABEL_W + c * THUMB
            path = path_fn(sid, tile)
            try:
                im = Image.open(path).convert("RGB").resize((THUMB, THUMB))
                mosaic.paste(im, (x0, y0))
            except Exception:
                pass

    out_path = os.path.join(OUTPUT_DIR, "mosaic_combined_all_pipelines_r2.png")
    mosaic.save(out_path)
    logging.info(f"Saved combined R2 mosaic: {out_path} ({n} lenses x {n_cols} pipelines, {width}x{height}px)")


if __name__ == "__main__":
    main()
