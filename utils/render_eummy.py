"""eummy colour rendering (Schirmer, M.; pip install eummy).

Unlike the other renderers, eummy is a CLI that reads the whole tile
mosaic and extracts cutouts itself from RA/Dec, so it takes FITS paths and
positions rather than a cutout array.

    from utils.render_eummy import render_eummy_tile
    render_eummy_tile(iyjh_paths, [{"id": ..., "ra": ..., "dec": ...}],
                      "work/_tile_102019592", "eummy_out", saturate=2.5)
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from astropy.io import fits

_EUMMY_RE = re.compile(r"^.+_(-?\d+\.\d+)([+-]\d+\.\d+)\.png$")


def find_eummy_exe() -> str:
    exe = shutil.which("eummy")
    if exe:
        return exe
    # not on PATH: next to this python, or where `pip install --user` puts scripts
    import site
    for folder in (os.path.dirname(sys.executable), os.path.join(site.getuserbase(), "bin")):
        candidate = os.path.join(folder, "eummy")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not find 'eummy'. Install it with 'pip install eummy'.")


def render_eummy_tile(
    iyjh_paths: list[str],
    sources: list[dict],
    tile_dir: str,
    out_dir: str,
    *,
    cutout_arcsec: float = 10.1,
    blackwhite: tuple[float, float] | None = None,
    pivot: float | None = None,
    contrast: float | None = None,
    saturate: float | tuple[float, float] | None = None,
    scales: tuple[float, float, float, float] | None = None,
    fr: float | None = None,
    extra_args: list[str] = (),
    nthreads: int = 1,
) -> int:
    """Render cutouts for sources on one tile; writes ``{out_dir}/{id}.png``.

    Parameters
    ----------
    iyjh_paths : the tile's VIS, NIR-Y, NIR-J, NIR-H mosaic FITS paths.
    sources : dicts with ``id``, ``ra``, ``dec`` (degrees), all on this tile.
    tile_dir : scratch dir; the mosaics are symlinked in and eummy runs here.
    out_dir : where the renamed PNGs go.
    cutout_arcsec : cutout width (eummy's help calls this a radius; its code uses it as width).

    Rendering options (None = eummy's own default, shown in brackets):
    blackwhite [(-1.3, 7000)] : min/max thresholds in the linear J-band image.
    pivot [0.15]              : fraction of max used as dynamic-range compression pivot.
    contrast [auto]           : extra contrast curve (1.0 EWS, 1.6 EDS, 0 off).
    saturate [2.0]            : saturation factor (one value, or a*/b* pair).
    scales [(0.002039, 0.595, 1.0, 1.0985)] : per-band I, Y, J, H flux scaling.
    fr [0.3]                  : H-band blending fraction into luminosity.
    extra_args                : any other eummy CLI flags, e.g. ["--blendIY"]
                                (see ``eummy --help``).

    Returns the number of PNGs written.
    """
    tile_dir = os.path.abspath(tile_dir)   # eummy needs an absolute --path
    out_dir = os.path.abspath(out_dir)
    os.makedirs(tile_dir, exist_ok=True)
    for path in iyjh_paths:
        link = os.path.join(tile_dir, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(path, link)

    ra_arr = np.array([s["ra"] for s in sources], dtype=np.float64)
    dec_arr = np.array([s["dec"] for s in sources], dtype=np.float64)
    catalog_path = os.path.join(tile_dir, "cutout_catalog.fits")
    fits.BinTableHDU.from_columns([
        fits.Column(name="RA", format="D", array=ra_arr),
        fits.Column(name="DEC", format="D", array=dec_arr),
    ]).writeto(catalog_path, overwrite=True)

    cmd = [find_eummy_exe(), "--path", tile_dir, "--cutouts", catalog_path,
           f'{cutout_arcsec}"', "--nthreads", str(nthreads)]
    for flag, val in [("--blackwhite", blackwhite), ("--pivot", pivot),
                      ("--contrast", contrast), ("--saturate", saturate),
                      ("--scales", scales), ("--fr", fr)]:
        if val is not None:
            cmd += [flag] + [str(v) for v in np.atleast_1d(val)]
    cmd += list(extra_args)
    subprocess.run(cmd, check=True, capture_output=True)

    # eummy names outputs TILE{id}_{ra}{dec}.png; match back to sources by position
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for png_path in glob.glob(os.path.join(tile_dir, "TILE*_*.png")):
        match = _EUMMY_RE.match(os.path.basename(png_path))
        if not match:
            continue
        ra, dec = float(match.group(1)), float(match.group(2))
        dra = (ra_arr - ra) * np.cos(np.radians(dec))
        i = int(np.argmin(np.hypot(dra, dec_arr - dec)))
        os.replace(png_path, os.path.join(out_dir, f"{sources[i]['id']}.png"))
        n_written += 1
    return n_written
