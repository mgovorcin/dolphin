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


class TestPerDateSimilarities:
    """One similarity raster per acquisition date."""

    @staticmethod
    def _write(tmp_path, stack, dates):
        from osgeo import gdal

        files = []
        for arr, d in zip(stack, dates, strict=True):
            fname = str(tmp_path / f"{d}.slc.tif")
            ds = gdal.GetDriverByName("GTiff").Create(
                fname, arr.shape[1], arr.shape[0], 1, gdal.GDT_CFloat32
            )
            ds.GetRasterBand(1).WriteArray(arr.astype("complex64"))
            ds = None
            files.append(Path(fname))
        return files

    @staticmethod
    def _medians(files):
        return np.array(
            [np.nanmedian(load_gdal(f, masked=True).filled(np.nan)) for f in files]
        )

    @pytest.fixture
    def dates(self):
        return [f"2020{m:02d}01" for m in range(1, 8)]

    @pytest.fixture
    def coherent_stack(self):
        """Seven dates of a smooth ramp plus light noise: every epoch is good."""
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[:60, :60]
        phase = np.array([0.02 * t * (xx + yy) for t in range(7)])
        return np.exp(1j * (phase + 0.2 * rng.standard_normal(phase.shape)))

    def test_writes_one_raster_per_date(self, tmp_path, coherent_stack, dates):
        files = self._write(tmp_path, coherent_stack, dates)
        out = similarity.create_per_date_similarities(
            files,
            dates,
            tmp_path / "per_date",
            search_radius=3,
            block_shape=(64, 64),
            num_threads=1,
            add_overviews=False,
        )
        assert [p.name for p in out] == [f"similarity_{d}.tif" for d in dates]
        assert all(p.exists() for p in out)

    def test_reference_date_gets_no_raster(self, tmp_path, coherent_stack, dates):
        files = self._write(tmp_path, coherent_stack, dates)
        out = similarity.create_per_date_similarities(
            files,
            dates,
            tmp_path / "per_date",
            search_radius=3,
            block_shape=(64, 64),
            num_threads=1,
            add_overviews=False,
            reference_idx=0,
        )
        assert [p.name for p in out] == [f"similarity_{d}.tif" for d in dates[1:]]

    def test_a_decorrelated_epoch_is_isolated(self, tmp_path, coherent_stack, dates):
        """The point of the product: one bad date shows up in its own raster only."""
        rng = np.random.default_rng(1)
        stack = coherent_stack.copy()
        bad = 3
        stack[bad] = np.exp(1j * rng.uniform(-np.pi, np.pi, stack.shape[-2:]))
        files = self._write(tmp_path, stack, dates)

        out = similarity.create_per_date_similarities(
            files,
            dates,
            tmp_path / "per_date",
            search_radius=3,
            block_shape=(64, 64),
            num_threads=1,
            add_overviews=False,
        )
        medians = self._medians(out)
        good = np.delete(medians, bad)
        assert medians[bad] < good.min() - 0.2, f"{medians = }"
        # The remaining epochs are barely affected by their bad neighbor
        assert good.std() < 0.1, f"{good = }"

    def test_every_date_has_the_same_number_of_partners(
        self, tmp_path, coherent_stack, dates
    ):
        """No epoch is weighted differently, including the first and last."""
        files = self._write(tmp_path, coherent_stack, dates)
        out = similarity.create_per_date_similarities(
            files,
            dates,
            tmp_path / "per_date",
            search_radius=3,
            block_shape=(64, 64),
            num_threads=1,
            add_overviews=False,
        )
        assert self._medians(out).std() < 0.05

    def test_blockwise_result_matches_a_single_block(
        self, tmp_path, coherent_stack, dates
    ):
        files = self._write(tmp_path, coherent_stack, dates)
        kwargs = {"search_radius": 3, "num_threads": 1, "add_overviews": False}
        many = similarity.create_per_date_similarities(
            files, dates, tmp_path / "many", block_shape=(16, 16), **kwargs
        )
        one = similarity.create_per_date_similarities(
            files, dates, tmp_path / "one", block_shape=(64, 64), **kwargs
        )
        for a, b in zip(many, one, strict=True):
            np.testing.assert_allclose(
                load_gdal(a), load_gdal(b), atol=1e-6, equal_nan=True
            )

    def test_mismatched_date_count_is_rejected(self, tmp_path, coherent_stack, dates):
        files = self._write(tmp_path, coherent_stack, dates)
        with pytest.raises(ValueError, match="Got 7 files but 6 dates"):
            similarity.create_per_date_similarities(
                files, dates[:-1], tmp_path / "per_date"
            )

    def test_too_short_a_stack_is_rejected(self, tmp_path, coherent_stack, dates):
        files = self._write(tmp_path, coherent_stack[:2], dates[:2])
        with pytest.raises(ValueError, match="at least 3 dates"):
            similarity.create_per_date_similarities(
                files, dates[:2], tmp_path / "per_date"
            )
