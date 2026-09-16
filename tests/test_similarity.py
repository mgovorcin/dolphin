import numpy as np
import pytest

from dolphin import similarity

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


class TestPerDateSimilarity:
    """Per-epoch similarity: one raster per date, plus a summary.

    The stack estimate answers "is this pixel coherent overall"; these answer
    "which date is dragging it down", which is what picking a reference date
    or dropping an acquisition needs.
    """

    def date_strs(self, slc_date_list):
        return [d.strftime("%Y%m%d") for d in slc_date_list]

    def test_writes_one_raster_per_date_plus_a_summary(
        self, tmp_path, slc_file_list, slc_date_list
    ):
        dates = self.date_strs(slc_date_list)
        per_date, summary = similarity.create_similarity_per_date(
            slc_file_list[:4],
            output_folder=tmp_path,
            date_strs=dates[:4],
            output_name="similarity_avg.tif",
            search_radius=2,
            nearest_n=1,
        )

        assert [p.name for p in per_date] == [f"similarity_{d}.tif" for d in dates[:4]]
        assert all(p.exists() for p in per_date)
        assert summary.name == "similarity_avg.tif"
        assert summary.exists()

    def test_summary_is_the_mean_of_the_per_date_rasters(
        self, tmp_path, slc_file_list, slc_date_list
    ):
        """nanmean, not nanmedian -- the choice the workflow relies on."""
        from dolphin.io import load_gdal

        dates = self.date_strs(slc_date_list)
        per_date, summary = similarity.create_similarity_per_date(
            slc_file_list[:4],
            output_folder=tmp_path,
            date_strs=dates[:4],
            output_name="similarity_avg.tif",
            search_radius=2,
            nearest_n=1,
        )

        stack = np.stack(
            [load_gdal(f, masked=True).filled(np.nan) for f in per_date], axis=0
        )
        expected = np.nanmean(stack, axis=0).astype("float32")
        written = load_gdal(summary, masked=True).filled(np.nan)

        np.testing.assert_allclose(written, expected, rtol=1e-6, equal_nan=True)

    def test_an_existing_per_date_raster_is_left_alone(
        self, tmp_path, slc_file_list, slc_date_list
    ):
        """Reruns skip finished dates, so a resumed run does not redo them."""
        dates = self.date_strs(slc_date_list)
        args = dict(
            output_folder=tmp_path,
            date_strs=dates[:3],
            output_name="similarity_avg.tif",
            search_radius=2,
            nearest_n=1,
        )
        per_date, summary = similarity.create_similarity_per_date(
            slc_file_list[:3], **args
        )
        first = per_date[0]
        stamp = first.stat().st_mtime_ns
        summary.unlink()  # force the summary to be rebuilt

        per_date_again, _ = similarity.create_similarity_per_date(
            slc_file_list[:3], **args
        )

        assert per_date_again == per_date
        assert first.stat().st_mtime_ns == stamp

    def test_unknown_sim_type_is_rejected(
        self, tmp_path, slc_file_list, slc_date_list
    ):
        with pytest.raises(ValueError, match="sim_type"):
            similarity.create_similarity_per_date(
                slc_file_list[:2],
                output_folder=tmp_path,
                date_strs=self.date_strs(slc_date_list)[:2],
                output_name="similarity_avg.tif",
                sim_type="mean",  # not median/max
            )


class TestCreatePerDateSimilarities:
    """The ifg-based variant: which dates appear is read off the filenames."""

    def make_ifgs(self, tmp_path, slc_stack, date_strs):
        """Single-look ifgs named ``{ref}_{sec}.tif``, consecutive pairs."""
        from osgeo import gdal

        d = tmp_path / "ifgs"
        d.mkdir()
        files = []
        for i in range(len(date_strs) - 1):
            ifg = slc_stack[i] * np.conj(slc_stack[i + 1])
            fname = d / f"{date_strs[i]}_{date_strs[i + 1]}.tif"
            ds = gdal.GetDriverByName("GTiff").Create(
                str(fname), ifg.shape[-1], ifg.shape[-2], 1, gdal.GDT_CFloat32
            )
            ds.GetRasterBand(1).WriteArray(ifg)
            ds = None
            files.append(fname)
        return files

    def test_one_raster_per_date_in_the_network(
        self, tmp_path, slc_stack, slc_date_list
    ):
        dates = [d.strftime("%Y%m%d") for d in slc_date_list][:4]
        ifgs = self.make_ifgs(tmp_path, slc_stack, dates)

        out = similarity.create_per_date_similarities(
            ifgs, output_dir=tmp_path / "sim", search_radius=2, add_overviews=False
        )

        # Three consecutive pairs touch all four dates.
        assert sorted(p.name for p in out) == sorted(
            f"similarity_{d}.tif" for d in dates
        )
        assert all(p.exists() for p in out)

    def test_a_filename_without_two_dates_is_named_in_the_error(self, tmp_path):
        with pytest.raises(ValueError, match="Could not parse two dates"):
            similarity.create_per_date_similarities(
                [tmp_path / "20200101_interferogram.tif"],
                output_dir=tmp_path / "sim",
            )

    def test_unknown_sim_type_is_rejected(self, tmp_path, slc_stack, slc_date_list):
        dates = [d.strftime("%Y%m%d") for d in slc_date_list][:3]
        ifgs = self.make_ifgs(tmp_path, slc_stack, dates)

        with pytest.raises(ValueError, match="sim_type"):
            similarity.create_per_date_similarities(
                ifgs, output_dir=tmp_path / "sim", sim_type="mean"
            )
