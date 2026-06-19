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


def test_load_sources_right_ascension_declination(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, right_ascension=[33.9], declination=[-45.5], tile_index=[102018212])

    with patch.object(m, "TILE_CENTRES_CSV", None):
        df = m.load_sources(str(csv))

    assert df["ra"].iloc[0] == 33.9
    assert df["dec"].iloc[0] == -45.5


def test_load_sources_target_ra_dec(tmp_path):
    csv = tmp_path / "src.csv"
    _write_csv(csv, target_ra=[33.9], target_dec=[-45.5], tile_index=[102018212])

    with patch.object(m, "TILE_CENTRES_CSV", None):
        df = m.load_sources(str(csv))

    assert df["ra"].iloc[0] == 33.9
    assert df["dec"].iloc[0] == -45.5


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
            {},
        ))
    tile_id, n_az, n_em, bulk_counts, n_skip = result
    assert n_az == 0 and n_em == 0 and n_skip == 1


# ── _render_bulk_variant ─────────────────────────────────────────────────────

def _random_cutout(size=32, seed=42):
    """Return a (size, size) float32 array with realistic-ish flux values."""
    rng = np.random.default_rng(seed)
    return (rng.random((size, size), dtype=np.float32) * 100).clip(0.1)


class TestRenderBulkVariant:

    def test_gz_arcsinh_vis_y_shape_and_dtype(self):
        vis, y = _random_cutout(seed=1), _random_cutout(seed=2)
        rgb = m._render_bulk_variant("gz_arcsinh_vis_y", vis, y, None)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_gz_arcsinh_vis_only_shape_and_dtype(self):
        vis = _random_cutout()
        rgb = m._render_bulk_variant("gz_arcsinh_vis_only", vis, None, None)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
        # greyscale — all channels equal
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_gz_arcsinh_triple_shape_and_dtype(self):
        vis, y, j = _random_cutout(seed=1), _random_cutout(seed=2), _random_cutout(seed=3)
        rgb = m._render_bulk_variant("gz_arcsinh_triple", vis, y, j)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_sw_mtf_vis_only_shape_and_dtype(self):
        vis = _random_cutout()
        rgb = m._render_bulk_variant("sw_mtf_vis_only", vis, None, None)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])

    def test_sw_mtf_vis_y_shape_and_dtype(self):
        vis, y = _random_cutout(seed=1), _random_cutout(seed=2)
        rgb = m._render_bulk_variant("sw_mtf_vis_y", vis, y, None)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_sw_mtf_vis_y_j_shape_and_dtype(self):
        vis, y, j = _random_cutout(seed=1), _random_cutout(seed=2), _random_cutout(seed=3)
        rgb = m._render_bulk_variant("sw_mtf_vis_y_j", vis, y, j)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_unknown_variant_returns_none(self):
        vis = _random_cutout()
        assert m._render_bulk_variant("nonexistent_stretch", vis, None, None) is None

    def test_does_not_mutate_input(self):
        vis = _random_cutout(seed=1)
        y = _random_cutout(seed=2)
        vis_orig = vis.copy()
        y_orig = y.copy()
        m._render_bulk_variant("gz_arcsinh_vis_y", vis, y, None)
        np.testing.assert_array_equal(vis, vis_orig)
        np.testing.assert_array_equal(y, y_orig)


# ── render_bulk_euclid_tile ──────────────────────────────────────────────────

class TestRenderBulkEuclidTile:

    def test_creates_files_for_each_variant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "NAMING", "id")
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "jpg")
        iyjh = _fake_iyjh()
        wcs = _simple_wcs()
        sources = [{"id": "s1", "ra": 33.9, "dec": -45.5, "size_pixel": 32}]

        gz_dir = str(tmp_path / "gz")
        sw_dir = str(tmp_path / "sw")
        os.makedirs(gz_dir)
        os.makedirs(sw_dir)
        out_dirs = {"gz_arcsinh_vis_y": gz_dir, "sw_mtf_vis_y_j": sw_dir}

        counts = m.render_bulk_euclid_tile(iyjh, wcs, sources, out_dirs)

        assert counts["gz_arcsinh_vis_y"] == 1
        assert counts["sw_mtf_vis_y_j"] == 1
        assert os.path.exists(os.path.join(gz_dir, "s1.jpg"))
        assert os.path.exists(os.path.join(sw_dir, "s1.jpg"))

    def test_skips_existing_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "NAMING", "id")
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "jpg")
        iyjh = _fake_iyjh()
        wcs = _simple_wcs()
        sources = [{"id": "s1", "ra": 33.9, "dec": -45.5, "size_pixel": 32}]

        out_dir = str(tmp_path / "gz")
        os.makedirs(out_dir)
        out_dirs = {"gz_arcsinh_vis_y": out_dir}

        # first run creates the file
        m.render_bulk_euclid_tile(iyjh, wcs, sources, out_dirs)
        # second run should skip
        counts = m.render_bulk_euclid_tile(iyjh, wcs, sources, out_dirs)
        assert counts["gz_arcsinh_vis_y"] == 0

    def test_empty_out_dirs_returns_empty(self, monkeypatch):
        monkeypatch.setattr(m, "NAMING", "id")
        iyjh = _fake_iyjh()
        wcs = _simple_wcs()
        sources = [{"id": "s1", "ra": 33.9, "dec": -45.5, "size_pixel": 32}]
        counts = m.render_bulk_euclid_tile(iyjh, wcs, sources, {})
        assert counts == {}

    def test_multiple_sources(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "NAMING", "id")
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "jpg")
        iyjh = _fake_iyjh()
        wcs = _simple_wcs()
        sources = [
            {"id": "a", "ra": 33.9, "dec": -45.5, "size_pixel": 32},
            {"id": "b", "ra": 33.9, "dec": -45.5, "size_pixel": 32},
            {"id": "c", "ra": 33.9, "dec": -45.5, "size_pixel": 32},
        ]
        out_dir = str(tmp_path / "mtf")
        os.makedirs(out_dir)
        counts = m.render_bulk_euclid_tile(iyjh, wcs, sources,
                                           {"sw_mtf_vis_only": out_dir})
        assert counts["sw_mtf_vis_only"] == 3
        for name in ["a", "b", "c"]:
            assert os.path.exists(os.path.join(out_dir, f"{name}.jpg"))

    def test_png_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "NAMING", "id")
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "png")
        iyjh = _fake_iyjh()
        wcs = _simple_wcs()
        sources = [{"id": "s1", "ra": 33.9, "dec": -45.5, "size_pixel": 32}]
        out_dir = str(tmp_path / "gz")
        os.makedirs(out_dir)
        counts = m.render_bulk_euclid_tile(iyjh, wcs, sources,
                                           {"gz_arcsinh_vis_only": out_dir})
        assert counts["gz_arcsinh_vis_only"] == 1
        assert os.path.exists(os.path.join(out_dir, "s1.png"))

    def test_output_is_valid_image(self, tmp_path, monkeypatch):
        """Saved file should be openable by PIL and have the right dimensions."""
        monkeypatch.setattr(m, "NAMING", "id")
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "jpg")
        iyjh = _fake_iyjh()
        wcs = _simple_wcs()
        sources = [{"id": "s1", "ra": 33.9, "dec": -45.5, "size_pixel": 32}]
        out_dir = str(tmp_path / "gz")
        os.makedirs(out_dir)
        m.render_bulk_euclid_tile(iyjh, wcs, sources,
                                  {"gz_arcsinh_vis_y": out_dir})
        from PIL import Image
        img = Image.open(os.path.join(out_dir, "s1.jpg"))
        assert img.size == (32, 32)
        assert img.mode == "RGB"


# ── _load_fits_cutout ───────────────────────────────────────────────────────

def _write_cutout_fits(path, bands, size=32):
    """Write a multi-extension FITS cutout with len(bands) channels."""
    primary = fits.PrimaryHDU()
    hdus = [primary]
    rng = np.random.default_rng(0)
    for i, _ in enumerate(bands):
        data = (rng.random((size, size), dtype=np.float32) * 100).clip(0.1)
        hdu = fits.ImageHDU(data=data, name=f"CHANNEL_{i+1}")
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(str(path), overwrite=True)


class TestLoadFitsCutout:

    def test_four_band_iyjh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        p = tmp_path / "test.fits"
        _write_cutout_fits(p, ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        arr = m._load_fits_cutout(str(p))
        assert arr.shape == (4, 32, 32)
        assert arr.dtype == np.float32
        assert arr[0].sum() > 0  # VIS
        assert arr[1].sum() > 0  # NIR_Y
        assert arr[2].sum() > 0  # NIR_J
        assert arr[3].sum() > 0  # NIR_H

    def test_three_band_cutana_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["NIR_Y", "NIR_J", "VIS"])
        p = tmp_path / "test.fits"
        _write_cutout_fits(p, ["NIR_Y", "NIR_J", "VIS"])
        arr = m._load_fits_cutout(str(p))
        assert arr.shape == (4, 32, 32)
        assert arr[0].sum() > 0   # VIS mapped from CHANNEL_3
        assert arr[1].sum() > 0   # NIR_Y mapped from CHANNEL_1
        assert arr[2].sum() > 0   # NIR_J mapped from CHANNEL_2
        assert arr[3].sum() == 0  # NIR_H absent — zero-filled

    def test_single_band_vis_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS"])
        p = tmp_path / "test.fits"
        _write_cutout_fits(p, ["VIS"])
        arr = m._load_fits_cutout(str(p))
        assert arr.shape == (4, 32, 32)
        assert arr[0].sum() > 0   # VIS
        assert arr[1].sum() == 0  # zero-filled
        assert arr[2].sum() == 0
        assert arr[3].sum() == 0

    def test_empty_fits_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS"])
        p = tmp_path / "empty.fits"
        fits.PrimaryHDU().writeto(str(p), overwrite=True)
        assert m._load_fits_cutout(str(p)) is None


# ── _render_single_fits ─────────────────────────────────────────────────────

class TestRenderSingleFits:

    def test_azulero_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        monkeypatch.setattr(m, "ENABLE_AZULERO", True)
        monkeypatch.setattr(m, "ENABLE_BULK_EUCLID", False)
        monkeypatch.setattr(m, "AZULERO_FORMAT", "jpg")

        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        az_dir = tmp_path / "azulero"
        az_dir.mkdir()

        p = fits_dir / "my_cutout.fits"
        _write_cutout_fits(p, ["VIS", "NIR_Y", "NIR_J", "NIR_H"])

        stem, n_az, bc = m._render_single_fits((str(p), str(az_dir), {}))
        assert stem == "my_cutout"
        assert n_az == 1
        assert (az_dir / "my_cutout.jpg").exists()
        from PIL import Image
        img = Image.open(str(az_dir / "my_cutout.jpg"))
        assert img.mode == "RGB"

    def test_bulk_euclid_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        monkeypatch.setattr(m, "ENABLE_AZULERO", False)
        monkeypatch.setattr(m, "ENABLE_BULK_EUCLID", True)
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "jpg")

        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        gz_dir = tmp_path / "gz"
        gz_dir.mkdir()

        p = fits_dir / "src42.fits"
        _write_cutout_fits(p, ["VIS", "NIR_Y", "NIR_J", "NIR_H"])

        bulk_dirs = {"gz_arcsinh_vis_y": str(gz_dir)}
        stem, n_az, bc = m._render_single_fits((str(p), str(tmp_path / "az"), bulk_dirs))
        assert stem == "src42"
        assert n_az == 0
        assert bc["gz_arcsinh_vis_y"] == 1
        assert (gz_dir / "src42.jpg").exists()

    def test_skips_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        monkeypatch.setattr(m, "ENABLE_AZULERO", True)
        monkeypatch.setattr(m, "ENABLE_BULK_EUCLID", False)
        monkeypatch.setattr(m, "AZULERO_FORMAT", "jpg")

        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        az_dir = tmp_path / "azulero"
        az_dir.mkdir()
        (az_dir / "existing.jpg").write_bytes(b"\xff\xd8\xff")

        p = fits_dir / "existing.fits"
        _write_cutout_fits(p, ["VIS", "NIR_Y", "NIR_J", "NIR_H"])

        stem, n_az, _ = m._render_single_fits((str(p), str(az_dir), {}))
        assert n_az == 0

    def test_empty_fits_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS"])
        monkeypatch.setattr(m, "ENABLE_AZULERO", True)
        monkeypatch.setattr(m, "ENABLE_BULK_EUCLID", False)

        az_dir = tmp_path / "azulero"
        az_dir.mkdir()

        p = tmp_path / "empty.fits"
        fits.PrimaryHDU().writeto(str(p), overwrite=True)

        stem, n_az, bc = m._render_single_fits((str(p), str(az_dir), {}))
        assert n_az == 0


# ── render_fits_dir ─────────────────────────────────────────────────────────

class TestRenderFitsDir:

    def test_renders_all_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        monkeypatch.setattr(m, "ENABLE_AZULERO", True)
        monkeypatch.setattr(m, "ENABLE_EUMMY", False)
        monkeypatch.setattr(m, "ENABLE_BULK_EUCLID", True)
        monkeypatch.setattr(m, "BULK_EUCLID_OUTPUTS", ["gz_arcsinh_vis_y"])
        monkeypatch.setattr(m, "BULK_EUCLID_FORMAT", "jpg")
        monkeypatch.setattr(m, "AZULERO_FORMAT", "jpg")
        monkeypatch.setattr(m, "N_WORKERS", 1)
        monkeypatch.setattr(m, "PROGRESS_BAR", False)

        fits_dir = tmp_path / "fits_in"
        fits_dir.mkdir()
        out_dir = tmp_path / "cutouts"
        monkeypatch.setattr(m, "INPUT_FITS_DIR", str(fits_dir))
        monkeypatch.setattr(m, "OUTPUT_DIR", str(out_dir))

        for name in ["src_a", "src_b"]:
            _write_cutout_fits(fits_dir / f"{name}.fits",
                               ["VIS", "NIR_Y", "NIR_J", "NIR_H"])

        m.render_fits_dir()

        assert (out_dir / "azulero" / "src_a.jpg").exists()
        assert (out_dir / "azulero" / "src_b.jpg").exists()
        assert (out_dir / "gz_arcsinh_vis_y" / "src_a.jpg").exists()
        assert (out_dir / "gz_arcsinh_vis_y" / "src_b.jpg").exists()

    def test_eummy_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(m, "FITS_BAND_ORDER", ["VIS"])
        monkeypatch.setattr(m, "ENABLE_AZULERO", False)
        monkeypatch.setattr(m, "ENABLE_EUMMY", True)
        monkeypatch.setattr(m, "ENABLE_BULK_EUCLID", False)
        monkeypatch.setattr(m, "N_WORKERS", 1)
        monkeypatch.setattr(m, "PROGRESS_BAR", False)

        fits_dir = tmp_path / "fits_in"
        fits_dir.mkdir()
        out_dir = tmp_path / "cutouts"
        monkeypatch.setattr(m, "INPUT_FITS_DIR", str(fits_dir))
        monkeypatch.setattr(m, "OUTPUT_DIR", str(out_dir))

        _write_cutout_fits(fits_dir / "x.fits", ["VIS"])

        import logging
        with caplog.at_level(logging.WARNING):
            m.render_fits_dir()
        assert any("eummy" in r.message.lower() for r in caplog.records)
