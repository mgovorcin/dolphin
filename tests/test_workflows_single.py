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
