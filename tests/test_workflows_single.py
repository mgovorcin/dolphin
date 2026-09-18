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
