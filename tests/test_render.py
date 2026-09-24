"""Tests for euclid_cutouts.render — the importable library used by tutorial.ipynb.

Requires azulero>=2.0, STCI, and a bulk-euclid-cutouts clone to be importable
(same requirements as tests/test_make_colour_cutouts.py). Set BULK_EUCLID_ROOT
to override the default clone location.
"""

import inspect
import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_cutouts import (
    build_azulero_transform,
    load_fits_cutout,
    render_azulero,
    render_bulk_variant,
    render_cutout,
    render_fits_dir,
    render_stci,
)

BULK_ROOT = os.environ.get("BULK_EUCLID_ROOT", "/media/user/bulk-euclid-cutouts")


def _fake_iyjh(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((4, h, w), dtype=np.float32) * 100).clip(0.1)


def _write_cutout_fits(path, bands, size=32, seed=0):
    primary = fits.PrimaryHDU()
    hdus = [primary]
    rng = np.random.default_rng(seed)
    for i, _ in enumerate(bands):
        data = (rng.random((size, size), dtype=np.float32) * 100).clip(0.1)
        hdus.append(fits.ImageHDU(data=data, name=f"CHANNEL_{i + 1}"))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)


# ── render_azulero / render_stci ─────────────────────────────────────────────
# Regression guard: render.py used to accept a `cutana_root` kwarg that was
# dropped when azulero rendering was brought in-package (commit e19c566).
# tutorial.ipynb kept passing it for months before this was caught — these
# tests exercise the real current signature so that drift is caught here
# instead of at tutorial-execution time.

class TestRenderAzulero:

    def test_signature_has_no_cutana_root(self):
        assert "cutana_root" not in inspect.signature(render_azulero).parameters

    def test_shape_and_dtype(self):
        rgb = render_azulero(_fake_iyjh())
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_does_not_mutate_input(self):
        iyjh = _fake_iyjh()
        original = iyjh.copy()
        render_azulero(iyjh)
        np.testing.assert_array_equal(iyjh, original)

    def test_build_azulero_transform_defaults(self):
        t = build_azulero_transform()
        assert t is not None


class TestRenderStci:

    def test_signature_has_no_cutana_root(self):
        assert "cutana_root" not in inspect.signature(render_stci).parameters

    def test_shape_and_dtype(self):
        rgb = render_stci(_fake_iyjh())
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8


# ── render_cutout ────────────────────────────────────────────────────────────

class TestRenderCutout:

    def test_signature_has_no_cutana_root(self):
        assert "cutana_root" not in inspect.signature(render_cutout).parameters

    def test_default_renderer_is_azulero(self):
        out = render_cutout(_fake_iyjh())
        assert list(out.keys()) == ["azulero"]

    def test_azulero_and_stci(self):
        out = render_cutout(_fake_iyjh(), renderers=["azulero", "stci"])
        assert set(out.keys()) == {"azulero", "stci"}
        for rgb in out.values():
            assert rgb.shape == (32, 32, 3)
            assert rgb.dtype == np.uint8

    def test_bulk_euclid_variant(self):
        out = render_cutout(
            _fake_iyjh(),
            renderers=["bulk_euclid"],
            bulk_variants=["gz_arcsinh_vis_y"],
            bulk_euclid_root=BULK_ROOT,
        )
        assert list(out.keys()) == ["gz_arcsinh_vis_y"]
        assert out["gz_arcsinh_vis_y"].shape == (32, 32, 3)

    def test_render_bulk_variant_direct(self):
        vis = _fake_iyjh(seed=1)[0]
        y = _fake_iyjh(seed=2)[1]
        rgb = render_bulk_variant("gz_arcsinh_vis_y", vis, y, None,
                                  bulk_euclid_root=BULK_ROOT)
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_all_three_renderers_together(self):
        out = render_cutout(
            _fake_iyjh(),
            renderers=["azulero", "stci", "bulk_euclid"],
            bulk_variants=["sw_mtf_vis_y_j"],
            bulk_euclid_root=BULK_ROOT,
        )
        assert set(out.keys()) == {"azulero", "stci", "sw_mtf_vis_y_j"}


# ── render_bulk_variant ──────────────────────────────────────────────────────

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
        vis = _fake_iyjh(seed=1)[0]
        y = _fake_iyjh(seed=2)[0]
        j = _fake_iyjh(seed=3)[0]
        rgb = render_bulk_variant(variant, vis, y, j, bulk_euclid_root=BULK_ROOT)
        assert rgb is not None
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8

    def test_unknown_variant_returns_none(self):
        vis = _fake_iyjh()[0]
        assert render_bulk_variant("bogus", vis, bulk_euclid_root=BULK_ROOT) is None


# ── load_fits_cutout ─────────────────────────────────────────────────────────

class TestLoadFitsCutout:

    def test_four_band(self, tmp_path):
        p = tmp_path / "c.fits"
        _write_cutout_fits(p, ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        iyjh = load_fits_cutout(str(p), band_order=["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        assert iyjh.shape == (4, 32, 32)
        assert iyjh.dtype == np.float32
        assert all(iyjh[i].sum() > 0 for i in range(4))

    def test_three_band_cutana_order(self, tmp_path):
        p = tmp_path / "c.fits"
        _write_cutout_fits(p, ["NIR_Y", "NIR_J", "VIS"])
        iyjh = load_fits_cutout(str(p), band_order=["NIR_Y", "NIR_J", "VIS"])
        assert iyjh[0].sum() > 0   # VIS, from CHANNEL_3
        assert iyjh[1].sum() > 0   # NIR_Y, from CHANNEL_1
        assert iyjh[2].sum() > 0   # NIR_J, from CHANNEL_2
        assert iyjh[3].sum() == 0  # NIR_H absent, zero-filled

    def test_empty_fits_returns_none(self, tmp_path):
        p = tmp_path / "empty.fits"
        fits.PrimaryHDU().writeto(str(p), overwrite=True)
        assert load_fits_cutout(str(p)) is None


# ── render_fits_dir ──────────────────────────────────────────────────────────
# Regression guard: render_fits_dir used to write flat {output_dir}/{renderer}/
# {stem}.jpg. Commit baa95c8 changed this to a per-tile subfolder,
# {output_dir}/{renderer}/{tile_id}/{stem}.jpg (tile_id = stem split on the
# first "_"), to avoid NFS slowdowns from flat million-entry directories --
# but tutorial.ipynb's display code assumed the old flat layout and broke
# silently until the fix. These tests pin the real, current layout.

class TestRenderFitsDir:

    def test_writes_per_tile_subfolder(self, tmp_path):
        fits_dir = tmp_path / "fits_in"
        fits_dir.mkdir()
        _write_cutout_fits(fits_dir / "102018212_111.fits",
                           ["VIS", "NIR_Y", "NIR_J", "NIR_H"], seed=1)
        _write_cutout_fits(fits_dir / "102018212_222.fits",
                           ["VIS", "NIR_Y", "NIR_J", "NIR_H"], seed=2)

        out_dir = tmp_path / "cutouts"
        totals = render_fits_dir(
            input_dir=str(fits_dir),
            output_dir=str(out_dir),
            enable_azulero=True,
            enable_stci=False,
            enable_bulk_euclid=False,
            band_order=["VIS", "NIR_Y", "NIR_J", "NIR_H"],
            n_workers=1,
        )

        assert totals["azulero"] == 2
        # Per-tile subfolder, not flat.
        assert (out_dir / "azulero" / "102018212" / "102018212_111.jpg").exists()
        assert (out_dir / "azulero" / "102018212" / "102018212_222.jpg").exists()
        assert not (out_dir / "azulero" / "102018212_111.jpg").exists()

    def test_no_fits_files_returns_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        totals = render_fits_dir(input_dir=str(empty_dir), output_dir=str(tmp_path / "out"))
        assert totals == {}

    def test_skips_already_rendered(self, tmp_path):
        fits_dir = tmp_path / "fits_in"
        fits_dir.mkdir()
        _write_cutout_fits(fits_dir / "tileA_1.fits",
                           ["VIS", "NIR_Y", "NIR_J", "NIR_H"])
        out_dir = tmp_path / "cutouts"

        render_fits_dir(
            input_dir=str(fits_dir), output_dir=str(out_dir),
            enable_azulero=True, enable_stci=False, enable_bulk_euclid=False,
            band_order=["VIS", "NIR_Y", "NIR_J", "NIR_H"], n_workers=1,
        )
        # Second pass should skip the already-rendered file (0 new renders).
        totals = render_fits_dir(
            input_dir=str(fits_dir), output_dir=str(out_dir),
            enable_azulero=True, enable_stci=False, enable_bulk_euclid=False,
            band_order=["VIS", "NIR_Y", "NIR_J", "NIR_H"], n_workers=1,
        )
        assert totals.get("azulero", 0) == 0
