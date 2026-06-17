"""
Build tile_centres.csv by scanning Euclid MER tile directories.

For each tile in each release, reads the VIS BGSUB-MOSAIC FITS header to derive
the tile centre (RA/Dec) and half-size in degrees. Output columns:

    tile_index, ra, dec, half_size_deg, release, release_dir

Run:
    python build_tile_centres.py

Output is written to TILE_CENTRES_OUT (default: ../tile_centres.csv next to this script).
"""

import glob
import logging
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# (release_name, release_dir) pairs to scan, in priority order.
RELEASES = [
    ("DR1_R2", "/media/home/data/euclid_idr1/DR1/R2"),
    ("DR1_R1", "/media/home/data/euclid_idr1/DR1/R1"),
    ("Q1_R1",  "/media/home/data/euclid_q1/Q1_R1"),
]

TILE_CENTRES_OUT = os.path.join(os.path.dirname(__file__), "..", "tile_centres.csv")
# ── END CONFIG ────────────────────────────────────────────────────────────────


def tile_centre_from_fits(fits_path: str) -> tuple[float, float, float] | None:
    """Return (ra, dec, half_size_deg) from the WCS of a BGSUB-MOSAIC FITS file."""
    try:
        with fits.open(fits_path, memmap=True) as h:
            hdr = h[0].header
            ny, nx = h[0].data.shape
        w = WCS(hdr)
        ra,  dec  = w.pixel_to_world_values(nx / 2, ny / 2)
        ra0, dec0 = w.pixel_to_world_values(0,  0)
        ra1, dec1 = w.pixel_to_world_values(nx, ny)
        d_dec = abs(dec1 - dec0)
        d_ra  = abs((ra1 - ra0) * np.cos(np.radians(dec)))
        half  = max(d_dec, d_ra) / 2.0
        return float(ra), float(dec), float(half)
    except Exception as e:
        logging.warning("Could not read WCS from %s: %s", fits_path, e)
        return None


def scan_release(release_name: str, release_dir: str) -> list[dict]:
    """Return one row per tile found under release_dir/MER/."""
    mer_dir = os.path.join(release_dir, "MER")
    tile_dirs = [d for d in glob.glob(os.path.join(mer_dir, "*")) if os.path.isdir(d)]
    logging.info("%s: %d tile dirs found", release_name, len(tile_dirs))

    rows = []
    for tile_dir in tile_dirs:
        tile_index = int(os.path.basename(tile_dir))
        pattern = os.path.join(tile_dir, "VIS", f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile_index}-*.fits")
        matches = glob.glob(pattern)
        if not matches:
            logging.debug("No VIS FITS for tile %d in %s — skipping", tile_index, release_name)
            continue

        result = tile_centre_from_fits(matches[0])
        if result is None:
            continue
        ra, dec, half = result
        rows.append({
            "tile_index":   tile_index,
            "ra":           ra,
            "dec":          dec,
            "half_size_deg": half,
            "release":      release_name,
            "release_dir":  release_dir,
        })

    logging.info("%s: %d tiles with valid WCS", release_name, len(rows))
    return rows


def main():
    all_rows = []
    for release_name, release_dir in RELEASES:
        if not os.path.isdir(release_dir):
            logging.warning("Release dir not found, skipping: %s", release_dir)
            continue
        all_rows.extend(scan_release(release_name, release_dir))

    df = pd.DataFrame(all_rows)
    out = os.path.abspath(TILE_CENTRES_OUT)
    df.to_csv(out, index=False)
    logging.info("Wrote %d rows to %s", len(df), out)
    print(df["release"].value_counts().to_string())


if __name__ == "__main__":
    main()
