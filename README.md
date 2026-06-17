# Euclid Colour Cutout Pipeline

Produce azulero JPEG and eummy PNG colour cutouts for Euclid Q1 and DR1 sources.

## Quick start

1. Prepare a CSV with your sources (only `ra` and `dec` are required):

```csv
ra,dec
33.941,-45.5
52.093,-30.5
```

2. Edit the CONFIG block at the top of `scripts/make_colour_cutouts.py` — set
   `SOURCES_CSV` to your file and `RELEASE_DIRS` for Q1 or DR1.

3. Run:

```bash
python scripts/make_colour_cutouts.py
```

Output lands in `cutouts/azulero/` (JPEGs) and `cutouts/eummy/` (PNGs), flat.

## Input CSV columns

| Column | Required | Description |
|---|---|---|
| `ra` | yes | Right ascension (deg) |
| `dec` | yes | Declination (deg) |
| `id` | no | Output filename stem (defaults to row index) |
| `object_id` | no | MER catalog object ID; required when `NAMING="q1_slde"` |
| `tile_index` | no | Euclid tile ID. Auto-resolved via HEALPix map if absent |
| `size_pixel` | no | Cutout size in VIS pixels, per source |
| `size_arcsec` | no | Cutout size in arcsec (converted to pixels; `size_pixel` takes precedence) |

When `tile_index` is not provided, the pipeline uses the Euclid tiling HEALPix
map (`data/tile_index_map.v1.2.fits.gz`, order 13 nested) to resolve it
automatically.  When neither `size_pixel` nor `size_arcsec` are given, all
cutouts use `DEFAULT_CUTOUT_PIXELS` from the config (101 px by default).

## Output structure

```
cutouts/
  fits/        FITS cutouts (Cutana) — reserved, not yet wired
  azulero/     colour JPEGs (azulero stretch)
  eummy/       colour PNGs  (eummy stretch)
```

All three are flat (no per-tile subdirs), mirroring Cutana's default layout.

## File naming

Set `NAMING` in the CONFIG block:

| Value | Filename stem | Notes |
|---|---|---|
| `"id"` | value of the `id` column | Default. Falls back to row index |
| `"q1_slde"` | `{tile_index}_{object_id}` | Requires `object_id` column |
| `"cutana_default"` | `{id}_{ra:.6f}_{dec:.6f}` | Mirrors Cutana's built-in template |

## Switching between Q1 and DR1

In the CONFIG block, comment/uncomment the `RELEASE_DIRS` lines:

```python
# Q1 (R1 only):
RELEASE_DIRS = ["/media/home/data/euclid_q1/Q1_R1"]

# DR1 (R2 preferred, R1 fallback):
# RELEASE_DIRS = ["/media/home/data/euclid_idr1/DR1/R2",
#                 "/media/home/data/euclid_idr1/DR1/R1"]
```

When `tile_centres.csv` is present, the pipeline resolves the correct
`release_dir` per tile automatically — `RELEASE_DIRS` only serves as a
fallback for tiles not listed in the CSV.

## Auxiliary scripts

| Script | Purpose |
|---|---|
| `scripts/build_tile_centres.py` | Regenerate `tile_centres.csv` by scanning MER dirs and reading VIS WCS headers |
| `scripts/fits_path_utils.py` | Shared FITS path resolution (BGSUB-MOSAIC lookup with coverage-aware selection) |
| `scripts/make_q1_colour_cutouts.py` | Original Q1-only script (superseded by `make_colour_cutouts.py`) |

## Dependencies

- Python 3.12+
- `numpy`, `pandas`, `astropy`, `Pillow`
- `azulero` 2.0 (installed from `/media/home/my_workspace/azulero` or pip)
- `eummy` (pip install)
- `healpy` (for tile lookup when `tile_index` is absent from the CSV)
- `cutana` (pip install, for FITS cutout mode — not yet wired)

The `azulero_render` module is imported from
`astronomaly-euclid/cutana_datalabs/`; make sure that repo is checked out at
`/media/user/astronomaly-euclid`.

## Tests

```bash
python -m pytest tests/ -v
```
