import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import make_q1_colour_cutouts as m


def test_parse_source_id_positive():
    tile, obj = m.parse_source_id("102018212_584242932512492979")
    assert tile == 102018212
    assert obj == 584242932512492979


def test_parse_source_id_negative():
    tile, obj = m.parse_source_id("102018212_NEG584242932512492979")
    assert tile == 102018212
    assert obj == -584242932512492979


def test_load_sources(tmp_path):
    import numpy as np
    feat = pd.DataFrame(
        {"feat_0": [1.0, 2.0]},
        index=pd.Index(
            ["102000001_100", "102000001_NEG200"],
            name="SourceID"
        )
    )
    pq = tmp_path / "features.parquet"
    feat.to_parquet(pq)

    coords = pd.DataFrame({
        "object_id": [100, -200],
        "right_ascension": [10.0, 11.0],
        "declination": [-27.0, -28.0],
    })
    csv = tmp_path / "coords.csv"
    coords.to_csv(csv, index=False)

    df = m.load_sources(str(pq), str(csv))
    assert len(df) == 2
    assert set(df.columns) == {"source_id", "tile_id", "ra", "dec"}

    row = df[df["source_id"] == "102000001_100"].iloc[0]
    assert row["tile_id"] == 102000001
    assert row["ra"] == pytest.approx(10.0)

    row2 = df[df["source_id"] == "102000001_NEG200"].iloc[0]
    assert row2["ra"] == pytest.approx(11.0)


def test_load_sources_drops_unmatched(tmp_path):
    feat = pd.DataFrame(
        {"feat_0": [1.0, 2.0, 3.0]},
        index=pd.Index(
            ["102000001_100", "102000001_NEG200", "102000001_300"],
            name="SourceID"
        )
    )
    pq = tmp_path / "features.parquet"
    feat.to_parquet(pq)

    coords = pd.DataFrame({
        "object_id": [100, -200],  # 300 has no match
        "right_ascension": [10.0, 11.0],
        "declination": [-27.0, -28.0],
    })
    csv = tmp_path / "coords.csv"
    coords.to_csv(csv, index=False)

    df = m.load_sources(str(pq), str(csv))
    assert len(df) == 2
    assert 300 not in df["source_id"].values


def test_resolve_iyjh_paths_returns_none_when_missing():
    with patch("make_q1_colour_cutouts.find_fits_paths_any_release", return_value=None):
        result = m.resolve_iyjh_paths(102000001, 10.0, -27.0)
    assert result is None


def test_resolve_iyjh_paths_returns_none_when_nirh_missing(tmp_path):
    vis  = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-VIS_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    niry = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    nirj = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    for p in [vis, niry, nirj]:
        Path(p).touch()

    with patch("make_q1_colour_cutouts.find_fits_paths_any_release", return_value=[vis, niry, nirj]):
        # find_iyjh_paths globs for NIR-H; no real file → returns None
        result = m.resolve_iyjh_paths(102000001, 10.0, -27.0)
    assert result is None


def test_resolve_iyjh_paths_returns_4_paths(tmp_path):
    vis  = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-VIS_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    niry = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    nirj = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    nirh = str(tmp_path / "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE102000001-ABC_20240101T000000.0Z_v1.fits")
    for p in [vis, niry, nirj, nirh]:
        Path(p).touch()

    with patch("make_q1_colour_cutouts.find_fits_paths_any_release", return_value=[vis, niry, nirj]):
        result = m.resolve_iyjh_paths(102000001, 10.0, -27.0)
    assert result is not None
    assert len(result) == 4
    assert any("NIR-H" in p for p in result)
    assert "VIS" in result[0]
    assert "NIR-H" in result[3]
