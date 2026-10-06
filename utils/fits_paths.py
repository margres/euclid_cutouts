"""Locate a tile's per-band BGSUB mosaic FITS files."""

import glob
import re

from astropy.io import fits
from astropy.wcs import WCS


def pick_best_coverage(matches: list[str], ra: float, dec: float) -> str:
    if len(matches) == 1:
        return matches[0]
    by_time = sorted(matches, key=lambda f: re.search(r"_(\d{8}T\d{6})", f).group(1))
    has_data = []
    for path in by_time:
        try:
            with fits.open(path, memmap=True) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                x, y = wcs.world_to_pixel_values(ra, dec)
                x, y = int(round(float(x))), int(round(float(y)))
                has_data.append(data[y, x] != 0)
        except Exception:
            has_data.append(False)
    for path, ok in zip(reversed(by_time), reversed(has_data)):
        if ok:
            return path
    return by_time[-1]


def find_fits_paths(tile_index: int, bands: list[str], release_dir: str,
                    band_to_instrument: dict[str, str],
                    ra: float, dec: float) -> list[str] | None:
    paths = []
    for band in bands:
        instrument = band_to_instrument[band]
        band_hyphen = band.replace("_", "-")
        pattern = (f"{release_dir}/MER/{tile_index}/{instrument}/"
                  f"EUC_MER_BGSUB-MOSAIC-{band_hyphen}_TILE{tile_index}-*.fits")
        matches = glob.glob(pattern)
        if not matches:
            return None
        paths.append(pick_best_coverage(matches, ra, dec))
    return paths


BAND_TO_INST = {"VIS": "VIS", "NIR_Y": "NISP", "NIR_J": "NISP", "NIR_H": "NISP"}
