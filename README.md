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

Only RA and Dec are required. Everything else is optional — the pipeline
fills in sensible defaults.

| Column | Required | Default | Description |
|---|---|---|---|
| RA | **yes** | — | Right ascension (deg) |
| Dec | **yes** | — | Declination (deg) |
| `id` | optional | row index | Output filename stem |
| `object_id` | optional | — | MER catalog object ID; only needed when `NAMING="q1_slde"` |
| `tile_index` | optional | HEALPix lookup | Euclid tile ID. Providing it avoids the need for the HEALPix map file, but does not speed up the run — the bottleneck is loading tile FITS data, not the lookup |
| `size_pixel` | optional | `DEFAULT_CUTOUT_PIXELS` (101) | Cutout size in VIS pixels, per source |
| `size_arcsec` | optional | — | Cutout size in arcsec (converted to pixels at 0.1"/px; `size_pixel` takes precedence) |

**RA/Dec column names** are auto-detected (case-insensitive):

| RA | Dec |
|---|---|
| `ra` | `dec` |
| `right_ascension` | `declination` |
| `target_ra` | `target_dec` |

If your CSV uses a different name, set `RA_COL` and `DEC_COL` in the CONFIG
block to override auto-detection.

When `tile_index` is not provided, the pipeline uses the Euclid tiling HEALPix
map (`data/tile_index_map.v1.2.fits.gz`, order 13 nested) to resolve it
automatically.

## Output structure

```
cutouts/
  azulero/            colour JPEGs (azulero stretch)
  eummy/              colour PNGs  (eummy stretch)
  gz_arcsinh_vis_y/   GZ arcsinh VIS+Y (bulk_euclid)
  sw_mtf_vis_y_j/     SW MTF VIS+Y+J   (bulk_euclid)
  ...                 one dir per BULK_EUCLID_OUTPUTS entry
```

All are flat (no per-tile subdirs), mirroring Cutana's default layout.

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

## Colouring pipelines

The cutout extraction and colour rendering rely on four tools developed
within (or for) the Euclid Consortium. Each renderer can be toggled
independently via the `ENABLE_*` flags in the CONFIG block:

### Cutana

[Cutana](https://github.com/esa/Cutana) (ESA;
[arXiv:2511.04429](https://arxiv.org/abs/2511.04429)) is a high-performance
Python pipeline for creating astronomical image cutouts from large FITS
datasets. It provides an interactive Jupyter UI and a programmatic
`Orchestrator` API with dynamic memory management, intelligent load balancing,
multi-channel FITS processing, flux-conserved resizing via drizzle, and WCS
preservation. This repository uses Cutana both directly (via
`cutana_cutouts.ipynb`) and indirectly — `make_colour_cutouts.py` reads the
same per-tile FITS stacks that Cutana resolves.

- Input: source catalogue (SourceID, RA, Dec, diameter, FITS paths) + MER tile stacks
- Output: multi-band FITS cutouts (or ZARR), with optional interactive UI
- Install: `pip install cutana`

### azulero

[azulero](https://doi.org/10.24400/815952/Azulero) (Basset et al.) is CNES's
colour-rendering package for Euclid tiles. It combines the four MER stack bands
(VIS, NIR-Y, NIR-J, NIR-H) into an LRGB composite using an asinh stretch,
per-band sharpening, dead-pixel inpainting, and configurable hue/saturation
mapping. This pipeline uses azulero's Python API (`azulero.image.color`,
`azulero.image.mask`) through the `azulero_render` wrapper in
`astronomaly-euclid/cutana_datalabs/`, which renders individual cutouts
in-memory rather than processing whole tiles via the `azul process` CLI.

- Input: 4-band IYJH (VIS, NIR-Y, NIR-J, NIR-H) cutout arrays
- Output: JPEG colour images (one per source)
- Install: `pip install azulero`

### eummy

[eummy](https://github.com/schirmermischa/eummy) (Schirmer, M.) creates colour
images from Euclid MER stacks using a contrast-based rendering with
configurable black/white thresholds, per-band scaling, and optional
colour-vision-deficiency simulation. It operates as a CLI tool that processes
whole tiles, after which the pipeline matches and renames the output cutouts
to source IDs by RA/Dec proximity.

- Input: 4-band FITS stacks (I, Y, J, H) on disk
- Output: PNG colour images (one per source, extracted from tile-level output)
- Install: `pip install eummy`

### bulk_euclid (arcsinh + MTF)

[bulk-euclid-cutouts](https://github.com/mwalmsley/bulk-euclid-cutouts)
(Walmsley, M.) provides two families of colour stretch:

- **Arcsinh** — per-band asinh dynamic-range compression, VIS+NISP compositing,
  and optional low-surface-brightness enhancement. Variants:
  `gz_arcsinh_vis_y` (VIS+Y), `gz_arcsinh_vis_only`,
  `gz_arcsinh_triple` (VIS+Y+J).
- **MTF** — midtone-transfer function (courtesy Tian Li): automatic
  "curves"-style contrast adjustment via a midtone balance parameter, with
  LAB-space luminosity replacement for multi-band composites. Variants:
  `sw_mtf_vis_only`, `sw_mtf_vis_y`, `sw_mtf_vis_y_j`.

Select which variants to produce by editing the `BULK_EUCLID_OUTPUTS` list in
the CONFIG block.

- Input: per-source VIS / NIR-Y / NIR-J cutout arrays (extracted from the same
  IYJH tile data used by azulero)
- Output: JPEG colour images (one per source per variant)
- Install: `pip install -e /path/to/bulk-euclid-cutouts` (or add its root to
  `_BULK_EUCLID_ROOT` in `make_colour_cutouts.py`)

## Dependencies

- Python 3.12+
- `numpy`, `pandas`, `astropy`, `Pillow`, `opencv-python`
- `azulero` 2.0 (`pip install azulero`)
- `eummy` (`pip install eummy`)
- `bulk-euclid-cutouts` (`pip install -e /path/to/bulk-euclid-cutouts`)
- `healpy` (for tile lookup when `tile_index` is absent from the CSV)
- `cutana` (`pip install cutana`)

The `azulero_render` module is imported from
`astronomaly-euclid/cutana_datalabs/`; make sure that repo is available and
its path is set in the `_CUTANA_ROOT` variable at the top of
`make_colour_cutouts.py`. The `bulk_euclid` package is imported from
`bulk-euclid-cutouts/`; set `_BULK_EUCLID_ROOT` accordingly.

## Tests

```bash
python -m pytest tests/ -v
```
