"""
Build a VIS-only Cutana catalogue for the 200k-object sample and run the cutout
pipeline, producing one VIS FITS cutout per source.

Run:
    python make_vis_only_cutouts_200k.py
"""

import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import make_cutana_cutouts as base

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CATALOGUE = "/media/user/cutana_dr1_pipeline/results/200k/cutana_output_200k/catalogue_200k.csv"
OUTPUT_DIR      = "/media/user/cutana_dr1_pipeline/results/200k/cutana_output_200k_vis"
BANDS           = ["VIS"]
TARGET_RESOLUTION = 101  # matches diameter_pixel in the input catalogue (no resampling)
# ── END CONFIG ────────────────────────────────────────────────────────────────


def build_vis_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tile_index"] = df["SourceID"].str.split("_").str[0].astype(int)

    logging.info(f"Resolving VIS-only FITS paths for {df['tile_index'].nunique()} unique tiles")
    tile_to_paths = {}
    for tile_index in df["tile_index"].unique():
        tile_to_paths[tile_index] = base.find_fits_paths_any_release(int(tile_index), BANDS)

    df["fits_file_paths"] = [
        json.dumps(tile_to_paths[t]) if tile_to_paths[t] is not None else None
        for t in df["tile_index"]
    ]

    missing = df["fits_file_paths"].isna().sum()
    df = df.dropna(subset=["fits_file_paths"])
    logging.info(f"{missing} sources dropped (missing VIS file for their tile). {len(df)} sources remain.")

    return df[["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"]].reset_index(drop=True)


def run_cutana(catalogue_path: str) -> None:
    from cutana import get_default_config, Orchestrator

    config = get_default_config()
    config.source_catalogue = catalogue_path
    config.output_dir = OUTPUT_DIR
    config.output_format = "fits"
    config.target_resolution = TARGET_RESOLUTION
    config.normalisation_method = "none"  # preserve raw flux values

    config.selected_extensions = [{"name": "VIS", "ext": "PrimaryHDU"}]
    config.channel_weights = {"VIS": [1.0]}

    logging.info(f"Running Cutana on {catalogue_path}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    orchestrator = Orchestrator(config)
    orchestrator.run()


def main():
    logging.info(f"Loading catalogue from {INPUT_CATALOGUE}")
    df = pd.read_csv(INPUT_CATALOGUE)
    logging.info(f"Loaded {len(df)} sources")

    catalogue = build_vis_catalogue(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    catalogue_path = os.path.join(OUTPUT_DIR, "vis_catalogue_200k.csv")
    catalogue.to_csv(catalogue_path, index=False)
    logging.info(f"Saved VIS-only catalogue to {catalogue_path}")

    print("\nFirst 3 rows of VIS-only catalogue:")
    print(catalogue.head(3).to_string())
    print()

    run_cutana(catalogue_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()
