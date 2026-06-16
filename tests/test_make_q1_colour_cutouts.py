import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

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
    assert set(df.columns) >= {"source_id", "tile_id", "ra", "dec"}
    assert df["tile_id"].iloc[0] == 102000001
    assert df["ra"].iloc[1] == pytest.approx(11.0)
