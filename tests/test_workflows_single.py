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
