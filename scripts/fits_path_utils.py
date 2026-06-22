"""
Shared FITS-path resolution helpers for the cutana/azulero/eummy cutout
pipelines (make_cutana_cutouts.py, make_azulero_cutouts.py,
make_eummy_cutouts.py).

Centralizes BGSUB-MOSAIC file lookup and coverage-aware selection between
multiple reprocessing runs of the same tile/band.
"""

import ast
import glob
import json
import logging
import re

import pandas as pd
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


def resolve_paths_by_tile(df: pd.DataFrame, bands: list[str], release_dirs: list[str],
                          band_to_instrument: dict[str, str],
                          tile_col: str = "tile_index") -> pd.DataFrame:
    """Add a ``fits_file_paths`` column by resolving FITS paths once per unique tile.

    Rows whose tile has no complete band set are dropped.

    Parameters
    ----------
    df : DataFrame with at least *tile_col*.
    bands : band names to resolve (e.g. ``["VIS", "NIR_Y", "NIR_J"]``).
    release_dirs : release directories to try in order.
    band_to_instrument : maps band name to instrument (``"VIS"`` or ``"NISP"``).
    tile_col : column containing the tile index.
    """
    logging.info("Resolving FITS paths once per unique tile")
    unique_tiles = df[tile_col].dropna().astype(int).unique()
    logging.info("Found %d unique tiles", len(unique_tiles))

    tile_to_paths: dict[int, str | None] = {}
    missing = 0
    for tile_index in unique_tiles:
        paths = find_fits_paths_any_release(
            int(tile_index), bands, release_dirs, band_to_instrument)
        if paths is None:
            tile_to_paths[int(tile_index)] = None
            missing += 1
        else:
            tile_to_paths[int(tile_index)] = json.dumps(paths)

    logging.info("%d tiles missing FITS files", missing)

    df = df.copy()
    df["fits_file_paths"] = df[tile_col].astype(int).map(tile_to_paths)
    n_dropped = df["fits_file_paths"].isna().sum()
    df = df.dropna(subset=["fits_file_paths"])
    logging.info("%d sources dropped (missing tiles). %d sources remain.",
                 n_dropped, len(df))
    return df


def find_iyjh_paths(fits_file_paths):
    """Given a 3-band fits_file_paths entry (VIS/NIR-Y/NIR-J), return 4
    IYJH paths [VIS, NIR-Y, NIR-J, NIR-H], deriving the missing NIR-H by
    globbing the same NISP tile directory. Returns None if NIR-H can't be
    found."""
    if isinstance(fits_file_paths, str):
        fits_file_paths = ast.literal_eval(fits_file_paths)

    _BAND_TAGS = ["VIS", "NIR-Y", "NIR-J", "NIR-H"]
    band_path = {}
    for path in fits_file_paths:
        for band in _BAND_TAGS:
            if f"-{band}_TILE" in path:
                band_path[band] = path
                break

    if "NIR-H" not in band_path:
        nisp_path = band_path.get("NIR-J") or band_path.get("NIR-Y")
        if nisp_path is None:
            return None
        match = re.search(r"-(NIR-[YJ])_(TILE\\d+)-", nisp_path)
        if match is None:
            return None
        pattern = nisp_path.replace(f"-{match.group(1)}_{match.group(2)}-",
                                     f"-NIR-H_{match.group(2)}-")
        pattern = re.sub(r"-[0-9A-F]+_\\d{8}T\\d{6}\\.\\d+Z_", "-*_*Z_", pattern)
        matches = glob.glob(pattern)
        if not matches:
            return None
        band_path["NIR-H"] = sorted(matches)[-1]

    if not all(band in band_path for band in _BAND_TAGS):
        return None
    return [band_path[band] for band in _BAND_TAGS]
