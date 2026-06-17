# Euclid Colour Cutout Pipeline

Produce azulero JPEG and eummy PNG colour cutouts for Euclid Q1 and DR1 sources.

## Quick start

1. Prepare a CSV with your sources (only `ra` and `dec` are required):

```csv
ra,dec
33.941125,-45.500000
52.093219,-30.499972
```

2. Edit the CONFIG block at the top of `scripts/make_colour_cutouts.py` — set
   `SOURCES_CSV` to your file and `RELEASE_DIRS` for Q1 or DR1.

3. Run:

```bash
python scripts/make_colour_cutouts.py
```

Output lands in `cutouts/azulero/` (JPEGs) and `cutouts/eummy/` (PNGs), flat.

## Input CSV columns

Only `ra` and `dec` are required. Everything else is optional — the pipeline
fills in sensible defaults.

| Column | Required | Default | Description |
|---|---|---|---|
| `ra` | **yes** | — | Right ascension (deg) |
| `dec` | **yes** | — | Declination (deg) |
| `id` | optional | row index | Output filename stem |
| `object_id` | optional | — | MER catalog object ID; only needed when `NAMING="q1_slde"` |
| `tile_index` | optional | HEALPix lookup | Euclid tile ID. Providing it avoids the need for the HEALPix map file, but does not speed up the run — the bottleneck is loading tile FITS data, not the lookup |
| `size_pixel` | optional | `DEFAULT_CUTOUT_PIXELS` (101) | Cutout size in VIS pixels, per source |
| `size_arcsec` | optional | — | Cutout size in arcsec (converted to pixels at 0.1"/px; `size_pixel` takes precedence) |

When `tile_index` is not provided, the pipeline uses the Euclid tiling HEALPix
map (`data/tile_index_map.v1.2.fits.gz`, order 13 nested) to resolve it
automatically.

## Output structure

```
cutouts/
  azulero/     colour JPEGs (azulero stretch)
  eummy/       colour PNGs  (eummy stretch)
```

Both are flat (no per-tile subdirs), mirroring Cutana's default layout.

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

## Repository contents

```
scripts/
  make_colour_cutouts.py        main pipeline (Q1 + DR1)
  make_q1_colour_cutouts.py     original Q1-only script (kept for provenance)
  build_tile_centres.py         regenerates tile_centres.csv from MER dirs
  fits_path_utils.py            shared FITS path resolution
tests/
  test_make_colour_cutouts.py   unit tests for the main pipeline
  test_make_q1_colour_cutouts.py
cutana_cutouts.ipynb            Cutana UI notebook for interactive cutouts
```

## Dependencies

- Python 3.12+
- `numpy`, `pandas`, `astropy`, `Pillow`
- `azulero` 2.0 (`pip install azulero`)
- `eummy` (`pip install eummy`)
- `healpy` (for tile lookup when `tile_index` is absent from the CSV)
- `cutana` (`pip install cutana`)

The `azulero_render` module is imported from
`astronomaly-euclid/cutana_datalabs/`; make sure that repo is available and
its path is set in the `_CUTANA_ROOT` variable at the top of
`make_colour_cutouts.py`.

## Tests

```bash
python -m pytest tests/ -v
```
