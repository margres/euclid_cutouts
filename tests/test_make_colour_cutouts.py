import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from unittest.mock import patch, MagicMock

import make_colour_cutouts as m


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_iyjh(ny=256, nx=256):
    rng = np.random.default_rng(0)
    return rng.random((4, ny, nx), dtype=np.float32)


def _simple_wcs(ra=33.9, dec=-45.5, scale=0.1 / 3600):
    w = WCS(naxis=2)
    w.wcs.crpix = [128, 128]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _write_csv(path, **cols):
    pd.DataFrame(cols).to_csv(path, index=False)


# ── load_sources ──────────────────────────────────────────────────────────────

def test_load_sources_full_columns(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv,
               id=["abc", "def"],
               ra=[33.9, 34.0], dec=[-45.5, -45.6],
               tile_index=[102018212, 102018212],
               size_pixel=[101, 50])

    with patch.object(m, "TILE_CENTRES_CSV", str(tmp_path / "nonexistent.csv")):
        df = m.load_sources(str(csv))

    assert list(df.columns) == ["id", "tile_index", "ra", "dec", "size_pixel", "release_dir"]
    assert df["id"].tolist() == ["abc", "def"]
    assert df["size_pixel"].tolist() == [101, 50]
    assert df["tile_index"].tolist() == [102018212, 102018212]


def test_load_sources_size_arcsec_conversion(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, ra=[33.9], dec=[-45.5], tile_index=[102018212], size_arcsec=[10.1])

    with patch.object(m, "TILE_CENTRES_CSV", None):
        df = m.load_sources(str(csv))

    expected = round(10.1 / m.VIS_ARCSEC_PER_PX)
    assert df["size_pixel"].iloc[0] == expected


def test_load_sources_default_size(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, ra=[33.9], dec=[-45.5], tile_index=[102018212])

    with patch.object(m, "TILE_CENTRES_CSV", None):
        df = m.load_sources(str(csv))

    assert df["size_pixel"].iloc[0] == m.DEFAULT_CUTOUT_PIXELS


def test_load_sources_default_id(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, ra=[33.9], dec=[-45.5], tile_index=[102018212])

    with patch.object(m, "TILE_CENTRES_CSV", None):
        df = m.load_sources(str(csv))

    assert df["id"].iloc[0] == "0"


def test_load_sources_healpix_lookup(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, ra=[33.9], dec=[-45.5])

    fake_df = pd.DataFrame({
        "ra": [33.9], "dec": [-45.5], "id": ["0"],
        "size_pixel": [101], "tile_index": [102018212], "release_dir": [None],
    })
    with patch.object(m, "_lookup_tile_indices", return_value=fake_df):
        df = m.load_sources(str(csv))

    assert df["tile_index"].iloc[0] == 102018212


def test_load_sources_missing_ra_dec_raises(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, object_id=[1])
    with pytest.raises(ValueError, match="ra.*dec"):
        m.load_sources(str(csv))


def test_load_sources_enriches_release_dir(tmp_path):
    tc = tmp_path / "tile_centres.csv"
    pd.DataFrame({
        "tile_index": [102018212],
        "release_dir": ["/data/DR1/R2"],
    }).to_csv(tc, index=False)

    csv = tmp_path / "src.csv"
    _write_csv(csv, ra=[33.9], dec=[-45.5], tile_index=[102018212])

    with patch.object(m, "TILE_CENTRES_CSV", str(tc)):
        df = m.load_sources(str(csv))

    assert df["release_dir"].iloc[0] == "/data/DR1/R2"


# ── _make_stem ────────────────────────────────────────────────────────────────

def test_make_stem_id(monkeypatch):
    monkeypatch.setattr(m, "NAMING", "id")
    src = {"id": "mysrc", "object_id": "999", "tile_index": 123, "ra": 1.0, "dec": -2.0}
    assert m._make_stem(src) == "mysrc"


def test_make_stem_q1_slde(monkeypatch):
    monkeypatch.setattr(m, "NAMING", "q1_slde")
    src = {"id": "mysrc", "object_id": "9876", "tile_index": 102018212, "ra": 1.0, "dec": -2.0}
    assert m._make_stem(src) == "102018212_9876"


def test_make_stem_cutana_default(monkeypatch):
    monkeypatch.setattr(m, "NAMING", "cutana_default")
    src = {"id": "mysrc", "object_id": "9876", "tile_index": 102018212, "ra": 33.941125, "dec": -45.5}
    assert m._make_stem(src) == "mysrc_33.941125_-45.500000"


# ── render_azulero_tile ───────────────────────────────────────────────────────

def test_render_azulero_tile_creates_flat_jpegs(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "NAMING", "id")
    iyjh = _fake_iyjh()
    wcs  = _simple_wcs()
    sources = [
        {"id": "src1", "ra": 33.9, "dec": -45.5, "size_pixel": 32},
        {"id": "src2", "ra": 33.9, "dec": -45.5, "size_pixel": 32},
    ]
    out = str(tmp_path)
    n = m.render_azulero_tile(iyjh, wcs, sources, out)
    assert n == 2
    # flat — no tile subdirectory
    assert (tmp_path / "src1.jpg").exists()
    assert (tmp_path / "src2.jpg").exists()
    assert not any(d.is_dir() for d in tmp_path.iterdir())


def test_render_azulero_tile_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "NAMING", "id")
    iyjh = _fake_iyjh()
    wcs  = _simple_wcs()
    src  = [{"id": "s1", "ra": 33.9, "dec": -45.5, "size_pixel": 32}]
    out  = str(tmp_path)
    m.render_azulero_tile(iyjh, wcs, src, out)
    n = m.render_azulero_tile(iyjh, wcs, src, out)
    assert n == 0


# ── rename_eummy_cutouts ──────────────────────────────────────────────────────

def test_rename_eummy_cutouts_uses_make_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "NAMING", "q1_slde")
    tile_dir = tmp_path / "ws"
    tile_dir.mkdir()
    out_dir  = tmp_path / "out"

    # Write a fake eummy PNG with the RA/Dec-encoded filename
    ra, dec = 33.941125, -45.5
    png = tile_dir / f"TILE102018212_{ra:.6f}{dec:+.6f}.png"
    png.write_bytes(b"\x89PNG\r\n")

    sources = [{"id": "mysrc", "object_id": "9876", "tile_index": 102018212,
                "ra": ra, "dec": dec}]
    n = m.rename_eummy_cutouts(str(tile_dir), sources, str(out_dir))

    assert n == 1
    assert (out_dir / "102018212_9876.png").exists()


# ── process_tile ──────────────────────────────────────────────────────────────

def test_process_tile_skips_missing_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "NAMING", "id")
    with patch.object(m, "resolve_iyjh_paths", return_value=None):
        result = m.process_tile((
            102018212,
            [{"id": "s1", "ra": 33.9, "dec": -45.5,
              "size_pixel": 101, "release_dir": None}],
            str(tmp_path / "az"),
            str(tmp_path / "em"),
        ))
    tile_id, n_az, n_em, n_skip = result
    assert n_az == 0 and n_em == 0 and n_skip == 1
