from pathlib import Path

import numpy as np
import pytest

from dolphin import similarity
from dolphin.io import load_gdal

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
)


def test_get_circle_idxs():
    idxs = similarity.get_circle_idxs(3)
    expected = np.array(
        [
            [-2, -1],
            [-2, 0],
            [-2, 1],
            [-1, -2],
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [-1, 2],
            [0, -2],
            [0, -1],
            [0, 1],
            [0, 2],
            [1, -2],
            [1, -1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, -1],
            [2, 0],
            [2, 1],
        ]
    )
    np.testing.assert_array_equal(idxs, expected)


def test_pixel_similarity():
    np.random.seed(12)
    phases1, phases2 = (
        np.random.uniform(2 * np.pi, size=(50,)),
        np.random.uniform(2 * np.pi, size=(50,)),
    )
    x1, x2 = np.exp(1j * phases1), np.exp(1j * phases2)
    small_sims = [similarity.phase_similarity(x1[:n], x2[:n]) for n in range(2, 6)]
    assert all(-1 <= sim <= 1 for sim in small_sims)

    # For random noise, the similarity should be closer to zero with more data
    sim_full = similarity.phase_similarity(x1, x2)
    assert np.abs(sim_full) < np.mean(np.abs(small_sims))

    # self similarity == 1
    assert similarity.phase_similarity(x1, x1) == 1


def test_pixel_similarity_zero_nan():
    x1, x2 = np.zeros((2, 10), dtype="complex64")
    sim = similarity.phase_similarity(x1, x2)
    assert sim == 0

    x1, x2 = np.full((2, 10), fill_value=np.nan + 1j * np.nan)
    sim = similarity.phase_similarity(x1, x2)
    assert np.isnan(sim)

    x1, x2 = np.ones((2, 10), dtype="complex64")
    sim = similarity.phase_similarity(x1, x2)
    assert sim == 1


def test_block_similarity_zero_nan():
    block_zeros = np.zeros((10, 4, 5), dtype="complex64")
    out = similarity.median_similarity(block_zeros, search_radius=2)
    assert np.isnan(out).all()

    block_nan = block_zeros * np.nan
    out = similarity.median_similarity(block_nan, search_radius=2)
    assert np.isnan(out).all()


class TestStackSimilarity:
    @pytest.fixture
    def ifg_stack(self, slc_stack):
        return slc_stack * slc_stack[[0]].conj()

    @pytest.mark.parametrize("radius", [2, 5, 9])
    @pytest.mark.parametrize("func", ["median", "max"])
    def test_basic(self, ifg_stack, radius, func):
        sim_func = getattr(similarity, f"{func}_similarity")
        sim = sim_func(ifg_stack, search_radius=radius)
        assert np.all(sim > -1)
        assert np.all(sim < 1)

    @pytest.mark.parametrize("radius", [2, 5, 9])
    @pytest.mark.parametrize("func", ["median", "max"])
    def test_max_similarity_masked(self, ifg_stack, radius, func):
        rows, cols = ifg_stack.shape[-2:]
        mask = np.random.rand(rows, cols).round().astype(bool)
        mask = np.array(
            [
                [True, False, False, True, False, False, False, False, True, True],
                [False, True, True, True, True, True, True, True, False, True],
                [True, False, False, False, False, True, True, False, True, False],
                [True, True, True, True, True, False, True, False, False, False],
                [False, False, True, True, True, True, True, True, False, True],
            ]
        )
        # Note: bottom right is surrounded by nans, so it will flip to a nan similarity

        sim_func = getattr(similarity, f"{func}_similarity")
        sim = sim_func(ifg_stack, search_radius=radius, mask=mask)

        assert ((np.nan_to_num(sim) > -1) & (np.nan_to_num(sim) < 1)).all()
        # The nan counts should be higher
        assert ~np.all(np.isnan(sim))
        assert np.isnan(sim).sum() >= (~mask).sum(), f"{sim = }, {mask = }"
        # The bottom right
        if radius == 2:
            assert np.isnan(sim[-1, -1])

    def test_create_similarity(self, tmp_path, slc_file_list):
        outfile = tmp_path / "med_sim.tif"
        similarity.create_similarities(
            slc_file_list, output_file=outfile, num_threads=1, block_shape=(64, 64)
        )


class TestZeroPhaseReferenceInflation:
    """Why the phase-linking reference must be dropped before computing similarity.

    A raster whose phase is identically zero is not an interferogram: it contributes
    ``cos(0) == 1`` to every pixel pair, adding a constant ``1 / n_ifgs`` to the sum.
    These tests pin the size of that bias so the workflow-level guard in
    `test_workflows_single.py` has a documented reason to exist.
    """

    @pytest.fixture
    def decorrelated_files(self, tmp_path):
        """Six rasters: a zero-phase reference followed by five of pure noise."""
        from osgeo import gdal

        rng = np.random.default_rng(42)
        shape = (80, 80)
        files = []
        for i in range(6):
            if i == 0:
                arr = np.ones(shape, dtype="complex64")
            else:
                arr = np.exp(1j * rng.uniform(-np.pi, np.pi, shape)).astype("complex64")
            fname = str(tmp_path / f"2020010{i + 1}.slc.tif")
            ds = gdal.GetDriverByName("GTiff").Create(
                fname, shape[1], shape[0], 1, gdal.GDT_CFloat32
            )
            ds.GetRasterBand(1).WriteArray(arr)
            ds = None
            files.append(Path(fname))
        return files

    def _median_similarity(self, files, out, nearest_n):
        similarity.create_similarities(
            files,
            output_file=out,
            num_threads=1,
            block_shape=(80, 80),
            search_radius=5,
            nearest_n=nearest_n,
            add_overviews=False,
        )
        return np.nanmedian(load_gdal(out, masked=True).filled(np.nan))

    @pytest.mark.parametrize("nearest_n", [None, 3])
    def test_decorrelated_data_scores_near_zero_without_the_reference(
        self, tmp_path, decorrelated_files, nearest_n
    ):
        """Pure noise must score ~0 once the zero-phase file is dropped."""
        got = self._median_similarity(
            decorrelated_files[1:], tmp_path / "without.tif", nearest_n
        )
        assert abs(got) < 0.05

    @pytest.mark.parametrize(
        ("nearest_n", "expected_ifgs"),
        [(None, 6), (3, 15)],
    )
    def test_including_the_reference_inflates_by_one_over_n(
        self, tmp_path, decorrelated_files, nearest_n, expected_ifgs
    ):
        """Keeping it biases pure noise up by about ``1 / n_ifgs``."""
        got = self._median_similarity(
            decorrelated_files, tmp_path / "with.tif", nearest_n
        )
        assert got == pytest.approx(1 / expected_ifgs, abs=0.04)
        assert got > 0.03, "the constant term should be plainly visible"
