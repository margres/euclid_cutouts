"""
Shared FITS-path resolution helpers for the cutana/azulero/eummy cutout
pipelines (make_cutana_cutouts.py, make_azulero_cutouts.py,
make_eummy_cutouts.py).

Centralizes BGSUB-MOSAIC file lookup and coverage-aware selection between
multiple reprocessing runs of the same tile/band.
"""

import glob
import logging
import re

from astropy.io import fits
from astropy.wcs import WCS


def pick_best_coverage(matches: list[str], ra: float | None, dec: float | None) -> str:
    """Pick which candidate mosaic file to use when a tile has multiple
    BGSUB-MOSAIC reprocessing runs.

    Filenames are `..._TILE{tile}-{HASH}_{timestamp}Z_...fits`: the random
    HASH precedes the timestamp, so `sorted(matches)[-1]` does NOT pick the
    chronologically newest file (and "newest" isn't even reliably "best" --
    different reprocessing campaigns aren't strictly monotonic in coverage).

    Instead: if ra/dec are given and there's more than one candidate, open
    each and check whether the pixel at (ra, dec) is non-zero. Prefer
    candidates with data there; among ties, prefer the chronologically
    newest (by the embedded timestamp). If none have data, fall back to the
    chronologically newest.
    """
    if len(matches) == 1:
        return matches[0]

    by_time = sorted(matches, key=lambda f: re.search(r"_(\d{8}T\d{6})", f).group(1))

    if ra is None or dec is None:
        return by_time[-1]

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
                     ra: float | None = None, dec: float | None = None) -> list[str] | None:
    """Return ordered list of BGSUB FITS paths for the given tile and bands in release_dir, or None if any is missing.

    If ra/dec are given and a tile has multiple reprocessing runs for a
    band, pick the one with non-zero coverage at (ra, dec) -- see
    pick_best_coverage.
    """
    paths = []
    for band in bands:
        instrument = band_to_instrument[band]
        band_hyphen = band.replace("_", "-")
        pattern = (
            f"{release_dir}/MER/{tile_index}/{instrument}/"
            f"EUC_MER_BGSUB-MOSAIC-{band_hyphen}_TILE{tile_index}-*.fits"
        )
        matches = glob.glob(pattern)
        if not matches:
            return None
        paths.append(pick_best_coverage(matches, ra, dec))
    return paths


def find_fits_paths_any_release(tile_index: int, bands: list[str], release_dirs: list[str],
                                 band_to_instrument: dict[str, str],
                                 ra: float | None = None, dec: float | None = None) -> list[str] | None:
    """Try each release dir in release_dirs order, returning the first complete set of FITS paths found."""
    for release_dir in release_dirs:
        paths = find_fits_paths(tile_index, bands, release_dir, band_to_instrument, ra, dec)
        if paths is not None:
            return paths
    logging.warning(f"  No complete set of {bands} files for tile {tile_index} in any release dir")
    return None
