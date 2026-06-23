"""
One-off: reorganize flat eummy PNGs into per-tile subfolders as JPG q99.

Reads all *.png from EUMMY_DIR (flat), parses tile_id from filename,
writes {tile_id}/{source_id}.jpg, deletes the original PNG.
"""

import glob
import logging
import os
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EUMMY_DIR = "/media/user/euclid_cutouts/results/q1_colour/eummy"
JPEG_QUALITY = 99


def main():
    pngs = glob.glob(os.path.join(EUMMY_DIR, "*.png"))
    logging.info("Found %d flat PNGs to reorganize", len(pngs))

    n_done = 0
    n_skip = 0
    for png_path in pngs:
        fname = os.path.basename(png_path)
        source_id = fname.removesuffix(".png")
        tile_id = source_id.split("_", 1)[0]

        tile_dir = os.path.join(EUMMY_DIR, tile_id)
        os.makedirs(tile_dir, exist_ok=True)
        dest = os.path.join(tile_dir, f"{source_id}.jpg")

        if os.path.exists(dest):
            os.remove(png_path)
            n_skip += 1
            continue

        try:
            Image.open(png_path).convert("RGB").save(dest, quality=JPEG_QUALITY)
            os.remove(png_path)
            n_done += 1
        except Exception as e:
            logging.warning("Bad file %s: %s — deleting", fname, e)
            os.remove(png_path)
            n_skip += 1

        if (n_done + n_skip) % 10000 == 0:
            logging.info("[%d/%d] converted=%d skipped=%d", n_done + n_skip, len(pngs), n_done, n_skip)

    logging.info("Done. converted=%d skipped=%d", n_done, n_skip)


if __name__ == "__main__":
    main()
