"""TL;DR: RA/Dec in, colour cutouts out, one folder per stretch.

    from utils.make_cutouts import make_cutouts
    make_cutouts("my_positions.csv")                              # all 8 stretches
    make_cutouts(df, stretches=["stci", "azulero"], out_dir="out")
    make_cutouts(df, stretches={"stci": {}, "azulero": {"saturation": 1.5}})

Writes  <out_dir>/<stretch>/<SourceID>.jpg  at JPEG quality 95 (+ <out_dir>/sources.csv with the
resolved tile / object id / match distance of every source).
Each stretch is a name; give a dict {name: renderer kwargs} to change its settings
(parameters: see the notebook, Step 4). Names:
    azulero, stci, gz_arcsinh_vis_y, gz_arcsinh_vis_only, gz_arcsinh_triple,
    sw_mtf_vis_only, sw_mtf_vis_y, sw_mtf_vis_y_j, eummy
Image format: fmt="jpg" (default, lossy, quality set by jpeg_quality=95) or
fmt="png" (lossless; jpeg_quality is then ignored).
eummy is whole-tile and slower, so it is not in the default set; it always
writes PNG. Re-running skips cutouts that already exist.
"""

import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from .fits_paths import find_fits_paths, BAND_TO_INST
from .render_azulero import render_azulero
from .render_bulk_euclid import render_bulk_euclid, VARIANTS as BULK_VARIANTS
from .render_stci import render_stci
from .render_eummy import render_eummy_tile
from .resolve_sources import resolve_sources

DEFAULT_STRETCHES = ["azulero", "stci"] + list(BULK_VARIANTS)
VIS_ARCSEC_PER_PX = 0.1


def _fetch(catalogue, bands, weights, size):
    from cutana import create_cutouts_direct, get_default_config
    catalogue = catalogue.copy()
    catalogue["fits_file_paths"] = [
        str(find_fits_paths(int(r.tile_index), bands, r.release_dir, BAND_TO_INST, ra=r.RA, dec=r.Dec))
        for r in catalogue.itertuples()
    ]
    catalogue["diameter_pixel"] = size
    config = get_default_config()
    config.target_resolution = size
    config.selected_extensions = [{"name": b.replace("_", "-"), "ext": "PrimaryHDU"} for b in bands]
    config.channel_weights = weights
    config.normalisation_method = "none"     # raw pixels, the renderers stretch
    config.apply_flux_conversion = False     # keep native ADU/s
    out = {}
    cols = ["SourceID", "RA", "Dec", "diameter_pixel", "fits_file_paths"]
    for batch in create_cutouts_direct(catalogue[cols], config):
        for cutout, meta in zip(batch["cutouts"], batch["metadata"]):
            out[meta["source_id"]] = cutout
    return out


def _render(name, iyjh, kwargs, bulk_root):
    if name == "azulero":
        return render_azulero(iyjh, **kwargs)
    if name == "stci":
        return render_stci(iyjh, **kwargs)
    return render_bulk_euclid(iyjh, name, bulk_euclid_root=bulk_root, **kwargs)


def make_cutouts(sources, stretches=None, out_dir="cutouts", size=101, fmt="jpg",
                 jpeg_quality=95, chunk=200, bulk_euclid_root=None, check=False, release=None):
    """sources: DataFrame or CSV path with RA/Dec columns; tile_index / release_dir /
    SourceID columns, if present, are trusted (check=True warns about a wrong tile).
    release: "q1" (the only release shipped here).
    Returns the resolved table."""
    stretches = DEFAULT_STRETCHES if stretches is None else stretches
    stretches = dict.fromkeys(stretches, {}) if not isinstance(stretches, dict) else stretches
    known = set(DEFAULT_STRETCHES) | {"eummy"}
    bad = [s for s in stretches if s not in known]
    if bad:
        raise ValueError(f"unknown stretch(es) {bad}; choose from {sorted(known)}")
    bulk_root = bulk_euclid_root or os.environ.get("BULK_EUCLID_ROOT", "bulk-euclid-cutouts")
    ext = lambda name: "png" if name == "eummy" else fmt

    table = resolve_sources(sources, check=check, release=release)
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "sources.csv"), index=False)
    path = lambda name, sid: os.path.join(out_dir, name, f"{sid}.{ext(name)}")
    for name in stretches:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    cutana_stretches = [s for s in stretches if s != "eummy"]
    todo = table[[not all(os.path.exists(path(s, sid)) for s in stretches)
                  for sid in table["SourceID"]]]
    print(f"{len(table)} sources resolved, {len(todo)} to make, stretches: {list(stretches)}")

    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        if cutana_stretches:
            vyj = _fetch(part, ["NIR_Y", "NIR_J", "VIS"],
                         {"NIR-Y": [1.0, 0, 0], "NIR-J": [0, 1.0, 0], "VIS": [0, 0, 1.0]}, size)
            nir_h = _fetch(part, ["NIR_H"], {"NIR-H": [1.0, 0, 0]}, size)
            for sid, raw in vyj.items():
                iyjh = np.zeros((4,) + raw.shape[:2], dtype=np.float32)
                iyjh[0], iyjh[1], iyjh[2] = raw[..., 2], raw[..., 0], raw[..., 1]
                iyjh[3] = nir_h[sid][..., 0]
                for name in cutana_stretches:
                    rgb = _render(name, iyjh, stretches[name], bulk_root)
                    kw = {"quality": jpeg_quality} if fmt == "jpg" else {}
                    Image.fromarray(rgb).save(path(name, sid), **kw)
        if "eummy" in stretches:
            for tile, grp in part.groupby("tile_index"):
                first = grp.iloc[0]
                paths = find_fits_paths(int(tile), ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                                        first.release_dir, BAND_TO_INST, ra=first.RA, dec=first.Dec)
                scratch = os.path.join(out_dir, "_eummy_scratch", str(tile))
                render_eummy_tile(paths, [{"id": r.SourceID, "ra": r.RA, "dec": r.Dec}
                                          for r in grp.itertuples()],
                                  scratch, os.path.join(out_dir, "eummy"),
                                  cutout_arcsec=size * VIS_ARCSEC_PER_PX, **stretches["eummy"])
            shutil.rmtree(os.path.join(out_dir, "_eummy_scratch"), ignore_errors=True)
        print(f"  {min(start + chunk, len(todo))}/{len(todo)}")
    return table
