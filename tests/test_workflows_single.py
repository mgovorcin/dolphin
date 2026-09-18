import pytest

from dolphin import stack
from dolphin.io import _readers
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
