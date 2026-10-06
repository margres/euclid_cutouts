# Euclid colour cutouts from RA/Dec

Give a list of sky positions (RA, Dec) and get colour cutouts of those sources
from the Euclid Q1 MER mosaics, rendered with several colour stretches, one
folder per stretch. Works on a machine where the Euclid Q1 data are mounted
(e.g. an ESA Datalabs workspace).

## Quick start

Open `cutana_to_colour_tutorial.ipynb`, run the setup cell (it installs what is
missing), then:

```python
import os
import pandas as pd
from utils.make_cutouts import make_cutouts

positions = pd.read_csv(os.path.join(HOWTO_DIR, "example_q1_lenses.csv")).head(5)
make_cutouts(positions, stretches=["stci", "azulero", "sw_mtf_vis_y_j"], out_dir="cutouts",
             fmt="jpg", jpeg_quality=95, release="q1")
```

You get `cutouts/<stretch>/<SourceID>.jpg` (JPEG quality 95; `fmt="png"` is
lossless) and `cutouts/sources.csv` with the resolved tile, object id and match
distance of every source.

## Input

A CSV or DataFrame with an RA and a Dec column (degrees): `ra`/`dec`,
`right_ascension`/`declination` or `target_ra`/`target_dec`, any case. Nothing
else is required. Optional columns used as given: `tile_index` (`tile_id`,
`tile`), `SourceID` (`source_id`, `id`), `release_dir`. Everything else is
carried through to `sources.csv`. The notebook documents the details.

## What is in here

- `cutana_to_colour_tutorial.ipynb`: the step-by-step tutorial.
- `utils/resolve_sources.py`: RA/Dec to tile, MER `OBJECT_ID` and `SourceID`.
- `utils/make_cutouts.py`: the one-call wrapper above.
- `utils/render_*.py`: the colour renderers (azulero, STCI, bulk_euclid, eummy).
- `example_outputs/`: what `make_cutouts` produces for three Q1 lenses, every stretch (one folder each, `sources.csv` has the resolved tile / object id).
- `example_q1_lenses.csv`: 20 grade-A Q1 lens candidates (Walmsley+25 and others), RA/Dec and name only.

## Tools used

[Cutana](https://github.com/esa/Cutana), [azulero](https://kabasset.github.io/azulero/v2.0.0/index.html),
STCI (courtesy Tian Li), [bulk-euclid-cutouts](https://github.com/mwalmsley/bulk-euclid-cutouts)
(Walmsley), [eummy](https://github.com/schirmermischa/eummy) (Schirmer).
The setup cell pins the versions these renderers were tested with
(azulero 2.0.0, STCI 0.1.1, eummy 1.3.2).

Notebook by Margherita Grespan.
