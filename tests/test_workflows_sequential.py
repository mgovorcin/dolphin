import numpy as np
import pytest

# from dolphin._types import HalfWindow, Strides
from dolphin.io import _readers, write_arr
from dolphin.phase_link import simulate
from dolphin.utils import compute_out_shape, gpu_is_available
from dolphin.workflows import sequential

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


def test_sequential_gtiff(tmp_path, slc_file_list):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    vrt_stack = _readers.VRTStack(slc_file_list, outfile=vrt_file)
    _, rows, cols = vrt_stack.shape

    half_window = {"x": cols // 2, "y": rows // 2}
    strides = {"x": 1, "y": 1}
    out_shape = compute_out_shape((rows, cols), strides=(strides["y"], strides["x"]))
    if not all(out_shape):
        pytest.skip(f"Output shape = {out_shape}")
    output_folder = tmp_path / "sequential"
    ms_size = 10

    sequential.run_wrapped_phase_sequential(
        slc_vrt_stack=vrt_stack,
        output_folder=output_folder,
        ministack_size=ms_size,
        half_window=half_window,
        strides=strides,
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
    )

    assert len(list(output_folder.glob("2*.slc.tif"))) == vrt_stack.shape[0]


# Input is only (5, 10) so we can't use a larger window.
@pytest.mark.parametrize(
    "half_window, strides",
    [
        ({"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 2, "y": 1}, {"x": 3, "y": 2}),
        ({"x": 3, "y": 1}, {"x": 2, "y": 1}),
        # (HalfWindow(1, 1), Strides(1, 1)),
        # (HalfWindow(1, 2), Strides(2, 3)),
        # (HalfWindow(1, 2), Strides(1, 3)),
    ],
)
def test_sequential_nc(tmp_path, slc_file_list_nc, half_window, strides):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    v = _readers.VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")

    _, rows, cols = v.shape
    out_shape = compute_out_shape((rows, cols), strides=(strides["y"], strides["x"]))
    if not all(out_shape):
        pytest.skip(f"Output shape = {out_shape}")

    sequential.run_wrapped_phase_sequential(
        slc_vrt_stack=v,
        output_folder=tmp_path / "sequential",
        ministack_size=10,
        half_window=half_window,
        strides=strides,
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
    )


@pytest.mark.parametrize("ministack_size", [5, 9, 20])
def test_sequential_ministack_sizes(tmp_path, slc_file_list_nc, ministack_size):
    """Check various strides/windows/ministacks with a NetCDF input stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    # Make it not a round number to test
    vrt_stack = _readers.VRTStack(
        slc_file_list_nc[:21], outfile=vrt_file, subdataset="data"
    )
    _, rows, cols = vrt_stack.shape

    # Record the warning, check after if it's thrown
    sequential.run_wrapped_phase_sequential(
        slc_vrt_stack=vrt_stack,
        ministack_size=ministack_size,
        output_folder=tmp_path / "sequential",
        half_window={"x": cols // 2, "y": rows // 2},
        strides={"x": 1, "y": 1},
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
    )


def test_per_date_similarity_across_ministacks(tmp_path, slc_file_list):
    """Per-date rasters stay in their ministack's subfolder and are all returned.

    They are a many-files-per-ministack output, so they follow `crlb/` and
    `closure_phases/` rather than the single-file outputs that get moved up.
    """
    vrt_stack = _readers.VRTStack(slc_file_list, outfile=tmp_path / "slc_stack.vrt")
    n_slc, rows, cols = vrt_stack.shape
    ms_size = 5
    assert n_slc > ms_size, "need more than one ministack for this test"

    output_folder = tmp_path / "sequential"
    result = sequential.run_wrapped_phase_sequential(
        slc_vrt_stack=vrt_stack,
        output_folder=output_folder,
        ministack_size=ms_size,
        half_window={"x": cols // 2, "y": rows // 2},
        strides={"x": 1, "y": 1},
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
        write_per_date_similarity=True,
    )
    per_date = result[-1]
    assert per_date, "no per-date rasters were collected"
    # Every one lives in a per_date_similarity subfolder of a ministack folder
    assert all(p.parent.name == "per_date_similarity" for p in per_date)
    assert all(p.exists() for p in per_date)
    # More than one ministack contributed
    assert len({p.parent.parent for p in per_date}) > 1
    # Every date is covered at most once, and the moved-up ministack rasters are
    # not confused with them
    names = [p.name for p in per_date]
    assert len(names) == len(set(names))
    assert not list(output_folder.glob("per_date_similarity"))


def test_full_stack_similarity_gets_the_water_mask(
    tmp_path, slc_file_list, monkeypatch
):
    """The multi-ministack raster takes the same water mask as the per-ministack ones.

    It is written only when a run spans several ministacks, and it is the one that
    ships (`stitched_similarity_files[-1]`). Left unmasked it would be the only
    similarity layer still scoring open water, which is exactly the layer a product
    would carry.
    """
    from dolphin import similarity

    vrt_stack = _readers.VRTStack(slc_file_list, outfile=tmp_path / "slc_stack.vrt")
    n_slc, rows, cols = vrt_stack.shape
    ms_size = 5
    assert n_slc > ms_size, "need more than one ministack for this test"

    mask_file = tmp_path / "water.tif"
    write_arr(
        arr=np.ones((rows, cols), dtype="uint8"),
        output_name=mask_file,
        like_filename=slc_file_list[0],
        dtype="uint8",
    )

    seen: list[object] = []
    real = similarity.create_similarities

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mask_file"))
        return real(*args, **kwargs)

    # `sequential` imported the name directly, so patch it there as well
    monkeypatch.setattr(similarity, "create_similarities", spy)
    monkeypatch.setattr(sequential, "create_similarities", spy)

    sequential.run_wrapped_phase_sequential(
        slc_vrt_stack=vrt_stack,
        output_folder=tmp_path / "sequential",
        ministack_size=ms_size,
        half_window={"x": cols // 2, "y": rows // 2},
        strides={"x": 1, "y": 1},
        ps_mask_file=None,
        amp_mean_file=None,
        amp_dispersion_file=None,
        shp_method="rect",
        shp_alpha=None,
        shp_nslc=None,
        similarity_mask_file=mask_file,
    )

    # One call per ministack, plus the full-stack one
    assert len(seen) > 2
    assert all(m is not None for m in seen), seen
