import os
import sys
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from euclid_cutouts.render import (
    load_fits_cutout,
    render_azulero,
    render_bulk_variant,
    render_cutout,
    render_fits_dir,
    IYJH_BAND_INDEX,
)

CUTANA_ROOT = "/media/user/astronomaly-euclid"
BULK_ROOT = "/media/user/bulk-euclid-cutouts"


def _fake_iyjh(size=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((4, size, size), dtype=np.float32) * 100).clip(0.1)


def _write_fits(path, bands, size=32):
    primary = fits.PrimaryHDU()
    hdus = [primary]
    rng = np.random.default_rng(0)
    for i in range(len(bands)):
        data = (rng.random((size, size), dtype=np.float32) * 100).clip(0.1)
        hdus.append(fits.ImageHDU(data=data, name=f"CHANNEL_{i+1}"))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)


# ── load_fits_cutout ────────────────────────────────────────────────────────

class TestLoadFitsCutout:

    def test_default_band_order(self, tmp_path):
        p = tmp_path / "test.fits"
        _write_fits(p, ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        arr = load_fits_cutout(str(p))
        assert arr.shape == (4, 32, 32)
        assert arr.dtype == np.float32

    def test_custom_band_order(self, tmp_path):
        p = tmp_path / "test.fits"
        _write_fits(p, ["NIR_Y", "NIR_J", "VIS"])
        arr = load_fits_cutout(str(p), band_order=["NIR_Y", "NIR_J", "VIS"])
        assert arr[0].sum() > 0   # VIS from CHANNEL_3
        assert arr[1].sum() > 0   # NIR_Y from CHANNEL_1
        assert arr[3].sum() == 0  # NIR_H zero-filled

    def test_empty_fits(self, tmp_path):
        p = tmp_path / "empty.fits"
        fits.PrimaryHDU().writeto(str(p))
        assert load_fits_cutout(str(p)) is None


# ── render_azulero ──────────────────────────────────────────────────────────

class TestRenderAzulero:

    def test_shape_and_dtype(self):
        iyjh = _fake_iyjh()
        rgb = render_azulero(iyjh, cutana_root=CUTANA_ROOT)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8


# ── render_bulk_variant ─────────────────────────────────────────────────────

class TestRenderBulkVariant:

    @pytest.mark.parametrize("variant", [
        "gz_arcsinh_vis_y",
        "gz_arcsinh_vis_only",
        "gz_arcsinh_triple",
        "sw_mtf_vis_only",
        "sw_mtf_vis_y",
        "sw_mtf_vis_y_j",
    ])
    def test_all_variants(self, variant):
        rng = np.random.default_rng(1)
        vis = (rng.random((32, 32), dtype=np.float32) * 100).clip(0.1)
        y = (rng.random((32, 32), dtype=np.float32) * 100).clip(0.1)
        j = (rng.random((32, 32), dtype=np.float32) * 100).clip(0.1)
        rgb = render_bulk_variant(variant, vis, y, j, bulk_euclid_root=BULK_ROOT)
        assert rgb is not None
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_unknown_returns_none(self):
        vis = np.ones((32, 32), dtype=np.float32)
        assert render_bulk_variant("bogus", vis, bulk_euclid_root=BULK_ROOT) is None


# ── render_cutout ───────────────────────────────────────────────────────────

class TestRenderCutout:

    def test_azulero_only(self):
        iyjh = _fake_iyjh()
        out = render_cutout(iyjh, renderers=["azulero"], cutana_root=CUTANA_ROOT)
        assert "azulero" in out
        assert out["azulero"].shape == (32, 32, 3)

    def test_bulk_only(self):
        iyjh = _fake_iyjh()
        out = render_cutout(iyjh, renderers=["bulk_euclid"],
                            bulk_variants=["gz_arcsinh_vis_y"],
                            bulk_euclid_root=BULK_ROOT)
        assert "gz_arcsinh_vis_y" in out
        assert "azulero" not in out

    def test_both(self):
        iyjh = _fake_iyjh()
        out = render_cutout(iyjh,
                            renderers=["azulero", "bulk_euclid"],
                            bulk_variants=["sw_mtf_vis_y_j"],
                            cutana_root=CUTANA_ROOT,
                            bulk_euclid_root=BULK_ROOT)
        assert "azulero" in out
        assert "sw_mtf_vis_y_j" in out


# ── render_fits_dir ─────────────────────────────────────────────────────────

class TestRenderFitsDir:

    def test_end_to_end(self, tmp_path):
        fits_dir = tmp_path / "fits_in"
        fits_dir.mkdir()
        out_dir = tmp_path / "out"

        for name in ["a", "b", "c"]:
            _write_fits(fits_dir / f"{name}.fits",
                        ["VIS", "NIR_Y", "NIR_J", "NIR_H"])

        totals = render_fits_dir(
            str(fits_dir), str(out_dir),
            enable_azulero=True,
            enable_bulk_euclid=True,
            bulk_variants=["gz_arcsinh_vis_y"],
            fmt="jpg",
            n_workers=1,
            cutana_root=CUTANA_ROOT,
            bulk_euclid_root=BULK_ROOT,
        )

        assert totals.get("azulero", 0) == 3
        assert totals.get("gz_arcsinh_vis_y", 0) == 3
        for name in ["a", "b", "c"]:
            assert (out_dir / "azulero" / f"{name}.jpg").exists()
            assert (out_dir / "gz_arcsinh_vis_y" / f"{name}.jpg").exists()

    def test_empty_dir(self, tmp_path):
        fits_dir = tmp_path / "empty"
        fits_dir.mkdir()
        totals = render_fits_dir(str(fits_dir), str(tmp_path / "out"))
        assert totals == {}
