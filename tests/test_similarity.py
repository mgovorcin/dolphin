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


class TestSimilarityMasking:
    """Masked pixels are skipped, as a centre pixel and as a neighbor.

    The motivating case is a coastline: decorrelated water inside the search radius
    otherwise drags down the similarity of the land pixels next to it.
    """

    @pytest.fixture
    def coast(self):
        """Left half coherent land, right half decorrelated water."""
        rng = np.random.default_rng(0)
        rows, cols, n_ifg, shore = 60, 90, 8, 45
        yy, xx = np.mgrid[:rows, :cols]
        phase = np.array([0.02 * k * (xx + yy) for k in range(n_ifg)])
        stack = np.exp(1j * (phase + 0.25 * rng.standard_normal(phase.shape)))
        stack[:, :, shore:] = np.exp(
            1j * rng.uniform(-np.pi, np.pi, (n_ifg, rows, cols - shore))
        )
        land = np.zeros((rows, cols), dtype=bool)
        land[:, :shore] = True
        return stack.astype("complex64"), land, shore

    def test_water_neighbors_depress_the_shoreline_unless_masked(self, coast):
        stack, land, shore = coast
        radius = 7
        unmasked = similarity.median_similarity(ifg_stack=stack, search_radius=radius)
        masked = similarity.median_similarity(
            ifg_stack=stack, search_radius=radius, mask=land
        )
        interior = slice(10, -10)
        inland = float(np.nanmean(masked[interior, : shore - 2 * radius]))
        at_shore_masked = float(np.nanmean(masked[interior, shore - 1]))
        at_shore_plain = float(np.nanmean(unmasked[interior, shore - 1]))

        # Masked, the shoreline reads like ordinary inland pixels...
        assert abs(at_shore_masked - inland) < 0.05, (at_shore_masked, inland)
        # ...unmasked, it is dragged well below them by the water in its disc
        assert inland - at_shore_plain > 0.1, (at_shore_plain, inland)
        # Beyond the search radius the two agree exactly
        np.testing.assert_allclose(
            masked[interior, : shore - radius],
            unmasked[interior, : shore - radius],
            atol=1e-6,
        )

    def test_masked_pixels_get_no_value(self, coast):
        stack, land, _ = coast
        out = similarity.median_similarity(ifg_stack=stack, search_radius=5, mask=land)
        assert np.isnan(out[~land]).all()
        assert not np.isnan(out[land]).all()

    def test_the_callers_mask_is_not_modified(self, coast):
        """The same mask array is reused for every block, so it must survive."""
        stack, land, _ = coast
        stack[:, :3, :3] = 0  # a patch of invalid data for the function to exclude
        before = land.copy()
        similarity.median_similarity(ifg_stack=stack, search_radius=5, mask=land)
        np.testing.assert_array_equal(land, before)

    def test_create_similarities_accepts_a_mask_file(self, tmp_path, coast):
        from osgeo import gdal

        stack, land, shore = coast
        files = []
        for i, layer in enumerate(stack):
            fname = str(tmp_path / f"2020010{i + 1}.slc.tif")
            ds = gdal.GetDriverByName("GTiff").Create(
                fname, layer.shape[1], layer.shape[0], 1, gdal.GDT_CFloat32
            )
            ds.GetRasterBand(1).WriteArray(layer)
            ds = None
            files.append(Path(fname))
        mask_file = tmp_path / "water_mask.tif"
        ds = gdal.GetDriverByName("GTiff").Create(
            str(mask_file), land.shape[1], land.shape[0], 1, gdal.GDT_Byte
        )
        ds.GetRasterBand(1).WriteArray(land.astype("uint8"))
        ds = None

        outs = {}
        for tag, mf in (("plain", None), ("masked", mask_file)):
            out = tmp_path / f"sim_{tag}.tif"
            similarity.create_similarities(
                files,
                output_file=out,
                search_radius=7,
                num_threads=1,
                block_shape=(64, 96),
                add_overviews=False,
                mask_file=mf,
            )
            outs[tag] = load_gdal(out, masked=True).filled(np.nan)

        interior = slice(10, -10)
        assert np.isnan(outs["masked"][~land]).all()
        # The file-level driver reproduces the in-memory behaviour
        assert (
            np.nanmean(outs["masked"][interior, shore - 1])
            - np.nanmean(outs["plain"][interior, shore - 1])
        ) > 0.1

    def test_a_mask_on_the_wrong_grid_is_rejected(self, tmp_path, coast):
        """The workflow's own mask_file is at input resolution; a strided run's
        interferograms are not, and reading one against the other would silently
        use the wrong corner of the mask."""
        from osgeo import gdal

        stack, land, _ = coast
        files = []
        for i, layer in enumerate(stack[:3]):
            fname = str(tmp_path / f"2020010{i + 1}.slc.tif")
            ds = gdal.GetDriverByName("GTiff").Create(
                fname, layer.shape[1], layer.shape[0], 1, gdal.GDT_CFloat32
            )
            ds.GetRasterBand(1).WriteArray(layer)
            ds = None
            files.append(Path(fname))
        # A mask twice the size, as an unstrided mask would be next to strided output
        big = tmp_path / "too_big.tif"
        ds = gdal.GetDriverByName("GTiff").Create(
            str(big), land.shape[1] * 2, land.shape[0] * 2, 1, gdal.GDT_Byte
        )
        ds.GetRasterBand(1).WriteArray(
            np.ones((land.shape[0] * 2, land.shape[1] * 2), "uint8")
        )
        ds = None

        with pytest.raises(ValueError, match="must be on the same grid"):
            similarity.create_similarities(
                files,
                output_file=tmp_path / "sim.tif",
                mask_file=big,
                num_threads=1,
                block_shape=(64, 96),
                add_overviews=False,
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
