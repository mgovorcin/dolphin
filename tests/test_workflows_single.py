from pathlib import Path

import numpy as np
import pytest

from dolphin import similarity, stack
from dolphin.io import _readers, load_gdal
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import single

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.mark.parametrize("write_extra", [False, True])
def test_sequential_gtiff(tmp_path, slc_file_list, write_extra: bool):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    is_compressed = [False] * len(files)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )

    hy, hx = 1, 2
    half_window = {"x": hx, "y": hy}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        shp_method="rect",
        write_crlb=write_extra,
        write_closure_phase=write_extra,
    )

    assert output_folder.exists()
    # Check that all the expected outputs are there
    assert len(list(output_folder.glob("2*.slc.tif"))) == 3
    assert len(list(output_folder.glob("compressed_*tif"))) == 1
    assert len(list(output_folder.glob("temporal_coherence*tif"))) == 1


def test_phase_linking_reference_excluded_from_similarity(
    tmp_path, slc_file_list, monkeypatch
):
    """The zero-phase reference SLC must not be passed to `create_similarities`.

    `create_similarities` treats each input as an interferogram against a reference
    date that is *not* in the list. When the phase-linking reference is itself one of
    the exported real SLCs its phase is identically zero, so including it contributes
    ``cos(0) == 1`` to every pixel pair and inflates the raster. Regression: the
    similarity raster's low tail was biased high by ~0.12 below the 5th percentile.
    """
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(files),
    )
    # With no compressed SLCs the output reference is the first exported SLC
    assert ministack.output_reference_idx == ministack.first_real_slc_idx == 0

    captured: dict[str, list] = {}

    def fake_create_similarities(ifg_file_list, *args, **kwargs):
        # `run_wrapped_phase_single` is wrapped in `atomic_output`, so read the
        # rasters here, while they still exist under the temporary folder.
        captured["names"] = [Path(f).name for f in ifg_file_list]
        captured["max_phase"] = [
            np.abs(np.angle(load_gdal(f))).max() for f in ifg_file_list
        ]

    monkeypatch.setattr(similarity, "create_similarities", fake_create_similarities)

    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=tmp_path / "single",
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
    )

    written = sorted(p.name for p in (tmp_path / "single").glob("2*.slc.tif"))
    # The reference is still exported as usual...
    assert len(written) == 3
    # ...but it is not one of the similarity inputs
    assert captured["names"] == written[1:]
    # and every file that *is* passed carries real phase
    assert all(mx > 0 for mx in captured["max_phase"])


def test_similarity_mask_file_reaches_create_similarities(
    tmp_path, slc_file_list, monkeypatch
):
    """The water mask is handed to the similarity step, and only to it.

    `mask_file` is the nodata mask -- CSLC footprints, bounds, layover/shadow -- and
    phase linking skips those blocks entirely. Water is different: it has signal, it
    is just decorrelated, so it refines the quality layer without stopping the
    estimate. The two therefore stay separate arguments.

    It is resolved with `stitching._get_matching_raster`, the helper
    `unwrapping.run` already uses, which warps onto the grid in use or passes the
    file through when it already matches. What the numbers then do is covered by
    `test_similarity.py::TestSimilarityMasking`; the fixture rasters here are far
    too small for a radius-7 disc to return anything.
    """
    from osgeo import gdal

    from dolphin.utils import compute_out_shape

    files = slc_file_list[:4]
    vrt_stack = _readers.VRTStack(files, outfile=tmp_path / "slc_stack.vrt")
    _, rows, cols = vrt_stack.shape
    strides = {"x": 2, "y": 2}
    out_shape = compute_out_shape((rows, cols), (strides["y"], strides["x"]))

    mask_file = tmp_path / "water.tif"
    ds = gdal.GetDriverByName("GTiff").Create(
        str(mask_file), out_shape[1], out_shape[0], 1, gdal.GDT_Byte
    )
    ds.GetRasterBand(1).WriteArray(np.ones(out_shape, dtype="uint8"))
    ds = None

    captured: dict[str, object] = {}

    def fake_create_similarities(*args, **kwargs):
        captured["mask_file"] = kwargs.get("mask_file")

    monkeypatch.setattr(similarity, "create_similarities", fake_create_similarities)

    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(files),
    )
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=tmp_path / "single",
        half_window={"x": 1, "y": 1},
        strides=strides,
        shp_method="rect",
        similarity_mask_file=mask_file,
    )
    passed = captured["mask_file"]
    assert passed is not None
    # Already on the similarity grid, so it is used as-is rather than re-warped
    assert load_gdal(passed).shape == out_shape


def test_no_similarity_mask_file_means_no_mask(tmp_path, slc_file_list, monkeypatch):
    """Leaving it unset must not pass the nodata `mask_file` in its place."""
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=tmp_path / "slc_stack.vrt")
    captured: dict[str, object] = {}

    def fake_create_similarities(*args, **kwargs):
        captured["mask_file"] = kwargs.get("mask_file")

    monkeypatch.setattr(similarity, "create_similarities", fake_create_similarities)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(files),
    )
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=tmp_path / "single",
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
    )
    assert captured["mask_file"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_per_date_similarity_is_opt_in(tmp_path, slc_file_list, enabled: bool):
    """The per-date rasters are written only when asked for, into their own folder."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:4]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(files),
    )
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
        write_per_date_similarity=enabled,
    )

    per_date = sorted((output_folder / "per_date_similarity").glob("similarity_*.tif"))
    if not enabled:
        assert not (output_folder / "per_date_similarity").exists()
        return
    # One per date, minus the zero-phase phase-linking reference
    assert len(per_date) == len(files) - 1
    # The subfolder must not be picked up by the ministack's own similarity glob
    assert len(list(output_folder.glob("similarity*"))) == 1


def test_compressed_slcs_get_no_per_date_similarity(tmp_path, slc_file_list):
    """Compressed SLCs are not acquisitions, so they get no per-date raster.

    They are excluded upstream -- only the real SLCs are written out as phase-linked
    outputs -- and with the reference being a compressed SLC none of the exported
    dates has zero phase, so every one of them gets a raster.
    """
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:4]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    dates = list(vrt_stack.dates)
    # Mark the first input as a compressed SLC, as a follow-on ministack would have
    dates[0] = [dates[0][0], dates[0][0]]
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=dates,
        is_compressed=[True] + [False] * (len(files) - 1),
    )
    assert ministack.first_real_slc_idx == 1
    assert ministack.output_reference_idx == 0

    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
        write_per_date_similarity=True,
    )

    per_date = sorted(
        p.name for p in (output_folder / "per_date_similarity").glob("similarity_*.tif")
    )
    real_dates = [d[0].strftime("%Y%m%d") for d in dates[1:]]
    assert per_date == [f"similarity_{d}.tif" for d in real_dates]
    # The compressed SLC's own date is not among them
    compressed_date = dates[0][0].strftime("%Y%m%d")
    assert f"similarity_{compressed_date}.tif" not in per_date
