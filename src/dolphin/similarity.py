"""Module for computing phase similarity between complex interferogram pixels.

Uses metric from [@Wang2022AccuratePersistentScatterer] for similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal, Sequence

import numba
import numpy as np
from numpy.typing import ArrayLike

from dolphin._types import PathOrStr

logger = logging.getLogger("dolphin")


@numba.njit(nogil=True)
def phase_similarity(x1: ArrayLike, x2: ArrayLike):
    """Compute the similarity between two complex 1D vectors."""
    n = len(x1)
    out = 0.0
    for i in range(n):
        out += np.real(x1[i] * np.conj(x2[i]))
    return out / n


def median_similarity(
    ifg_stack: ArrayLike, search_radius: int, mask: ArrayLike | None = None
):
    """Compute the median similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (5) of [@Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, or floating point phase.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
    mask: ArrayLike (optional)
        Array of mask from True/False indicating whether to include the pixel (True)
        or ignore it (False).

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the median similarity at each pixel.

    """
    return _create_loop_and_run(
        ifg_stack=ifg_stack,
        search_radius=search_radius,
        mask=mask,
        func=np.nanmedian,
    )


def max_similarity(
    ifg_stack: ArrayLike, search_radius: int, mask: ArrayLike | None = None
):
    """Compute the maximum similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (6) of [@Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, or floating point phase.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
    mask: ArrayLike (optional)
        Array of mask from True/False indicating whether to include the pixel (True)
        or ignore it (False).

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the maximum similarity for any neighbor
        at a pixel.

    """
    return _create_loop_and_run(
        ifg_stack=ifg_stack,
        search_radius=search_radius,
        mask=mask,
        func=np.nanmax,
    )


def _create_loop_and_run(
    ifg_stack: ArrayLike,
    search_radius: int,
    mask: ArrayLike | None,
    func: Callable[[ArrayLike], np.ndarray],
):
    _n_ifg, rows, cols = ifg_stack.shape
    # Mark any nans/all zeros as invalid
    invalid_mask = np.nan_to_num(ifg_stack).sum(axis=0) == 0
    if not np.iscomplexobj(ifg_stack):
        unit_ifgs = np.exp(1j * ifg_stack)
    else:
        unit_ifgs = np.exp(1j * np.angle(ifg_stack))
    out_similarity = np.full((rows, cols), fill_value=np.nan, dtype="float32")
    if mask is None:
        mask = np.ones((rows, cols), dtype="bool")
    mask[invalid_mask] = False

    if mask.shape != (rows, cols):
        raise ValueError(f"{ifg_stack.shape = }, but {mask.shape = }")

    idxs = get_circle_idxs(search_radius)
    loop_func = _make_loop_function(func)
    return loop_func(unit_ifgs, idxs, mask, out_similarity)


def _make_loop_function(
    summary_func: Callable[[ArrayLike], np.ndarray],
):
    """Create a JIT-ed function for some summary of the neighbors's similarity.

    E.g.: for median similarity, call

        median_sim = _make_loop_function(np.median)
    """

    @numba.njit(nogil=True, parallel=True)
    def _masked_sim_loop(
        ifg_stack: np.ndarray,
        idxs: np.ndarray,
        mask: np.ndarray,
        out_similarity: np.ndarray,
    ) -> np.ndarray:
        """Loop over each pixel, make a masked phase similarity to its neighbors."""
        _, rows, cols = ifg_stack.shape

        num_compare_pixels = len(idxs)
        # Buffer to hold all comparison during the parallel loop
        cur_sim = np.zeros((rows, cols, num_compare_pixels))

        for r0 in numba.prange(rows):
            for c0 in range(cols):
                # Get the current pixel
                m0 = mask[r0, c0]
                if not m0:
                    continue
                x0 = ifg_stack[:, r0, c0]

                cur_sim_vec = cur_sim[r0, c0]
                count = 0

                # compare to all pixels in the circle around it
                for i_idx in range(num_compare_pixels):
                    ir, ic = idxs[i_idx]
                    # Clip to the image bounds
                    r = max(min(r0 + ir, rows - 1), 0)
                    c = max(min(c0 + ic, cols - 1), 0)
                    if r == r0 and c == c0:
                        continue

                    # Check for a pixel to ignore
                    if not mask[r, c]:
                        continue

                    x = ifg_stack[:, r, c]
                    # cur_sim_vec[count] = w * phase_similarity(x0, x)
                    cur_sim_vec[count] = phase_similarity(x0, x)
                    count += 1
                    # Assuming `summary_func` is nan-aware
                if count > 0:  # a 0 count will fail for `max`
                    out_similarity[r0, c0] = summary_func(cur_sim_vec[:count])
        return out_similarity

    return _masked_sim_loop


def get_circle_idxs(
    max_radius: int, min_radius: int = 0, sort_output: bool = True
) -> np.ndarray:
    """Get the relative indices of neighboring pixels in a circle.

    Adapted from c++ version of `psps` package:
    https://github.com/UT-Radar-Interferometry-Group/psps/blob/a15d458817fe7d06a6edaa0b3208ea78bc4782e7/src/cpp/similarity.cpp#L16
    """
    # using the mid-point circle drawing algorithm to search for neighboring PS pixels
    # # code adapted from "https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/"
    visited = np.zeros((max_radius, max_radius), dtype=bool)
    visited[0][0] = True

    indices = []
    for r in range(1, max_radius):
        x = r
        y = 0
        p = 1 - r
        if r > min_radius:
            indices.append([r, 0])
            indices.append([-r, 0])
            indices.append([0, r])
            indices.append([0, -r])

        visited[r][0] = True
        visited[0][r] = True
        # flag > 0 means there are holes between concentric circles
        flag = 0
        while x > y:
            # do not need to fill holes
            if flag == 0:
                y += 1
                if p <= 0:
                    # Mid-point is inside or on the perimeter
                    p += 2 * y + 1
                else:
                    # Mid-point is outside the perimeter
                    x -= 1
                    p += 2 * y - 2 * x + 1

            else:
                flag -= 1

            # All the perimeter points have already been visited
            if x < y:
                break

            while not visited[x - 1][y]:
                x -= 1
                flag += 1

            visited[x][y] = True
            visited[y][x] = True
            if r > min_radius:
                indices.append([x, y])
                indices.append([-x, -y])
                indices.append([x, -y])
                indices.append([-x, y])

                if x != y:
                    indices.append([y, x])
                    indices.append([-y, -x])
                    indices.append([y, -x])
                    indices.append([-y, x])

            if flag > 0:
                x += 1

    if sort_output:
        # Sorting makes it run faster, better data access patterns
        return np.array(sorted(indices))
    else:
        # Indices run from middle outward
        return np.array(indices)


def create_similarities(
    ifg_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    search_radius: int = 7,
    sim_type: Literal["median", "max"] = "median",
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
    add_overviews: bool = True,
    nearest_n: int | None = None,
):
    """Create a similarity raster from as stack of ifg files.

    Parameters
    ----------
    ifg_file_list : Sequence[PathOrStr]
        Paths to input interferograms
    output_file : PathOrStr
        Output raster path
    search_radius : int, optional
        Maximum radius to search for pixels, by default 7
    sim_type : str, optional
        Type of similarity function to run, by default "median"
        Choices: "median", "max"
    block_shape : tuple[int, int], optional
        Size of blocks to process at one time from `ifg_file_list`
        by default (512, 512)
    num_threads : int, optional
        Number of parallel blocks to process, by default 5
    add_overviews : bool, optional
        Whether to create overviews in `output_file` by default True
    nearest_n : int, optional
        If provided, reform the nearest N interferograms before computing similarity.

    """
    from dolphin._overviews import Resampling, create_image_overviews
    from dolphin.io import BackgroundRasterWriter, VRTStack, process_blocks
    from dolphin.timeseries import get_incidence_matrix

    if Path(output_file).exists():
        logger.info(f"{output_file} exists, skipping")
        return

    if sim_type == "median":
        sim_function = median_similarity
    elif sim_type == "max":
        sim_function = max_similarity
    else:
        raise ValueError(f"Unrecognized {sim_type = }")

    nodata_block = np.full(block_shape, fill_value=np.nan, dtype="float32")

    if nearest_n is not None:
        incidence_matrix = get_incidence_matrix(
            _create_nearest_n_pairs(len(ifg_file_list) + 1, n=nearest_n)
        )
        assert incidence_matrix.shape[1] == len(ifg_file_list)
    else:
        incidence_matrix = None

    def calc_sim(readers, rows, cols):
        block = readers[0][:, rows, cols]
        if np.sum(block) == 0 or np.isnan(block).all():
            return nodata_block[rows, cols], rows, cols

        if incidence_matrix is not None:
            block = _calc_nearest_diffs(block, incidence_matrix)

        out_avg = sim_function(ifg_stack=block, search_radius=search_radius)
        logger.debug(f"{rows = }, {cols = }, {block.shape = }, {out_avg.shape = }")
        return out_avg, rows, cols

    out_dir = Path(output_file).parent
    reader = VRTStack(ifg_file_list, outfile=out_dir / "sim_inputs.vrt")

    writer = BackgroundRasterWriter(
        output_file,
        like_filename=ifg_file_list[0],
        dtype="float32",
        driver="GTiff",
        nodata=np.nan,
    )
    process_blocks(
        [reader],
        writer,
        func=calc_sim,
        block_shape=block_shape,
        overlaps=(search_radius, search_radius),
        num_threads=num_threads,
    )
    writer.notify_finished()

    if add_overviews:
        logger.info("Creating overviews for unwrapped images")
        create_image_overviews(Path(output_file), resampling=Resampling.AVERAGE)


def create_similarity_per_date(
    slc_file_list: Sequence[PathOrStr],
    output_folder: PathOrStr,
    date_strs: Sequence[str],
    output_name: str,
    search_radius: int = 7,
    nearest_n: int = 3,
    sim_type: Literal["median", "max"] = "median",
    block_shape: tuple[int, int] = (512, 512),  # noqa: ARG001
    num_threads: int = 1,  # noqa: ARG001
) -> tuple[list[Path], Path]:
    """Compute phase similarity for each date using nearest-N SLC neighbors.

    For date ``i``, forms nearest-N single-look interferograms
    (``slc[i] * conj(slc[j])`` for ``j`` in ``[i-n, ..., i+n]``) and
    computes spatial similarity on that subset.  Writes one
    ``similarity_{date}.tif`` per date, then averages them into a single
    ``{output_name}`` summary raster.

    Parameters
    ----------
    slc_file_list : Sequence[PathOrStr]
        Phase-linked SLC files in chronological order.
    output_folder : PathOrStr
        Directory to write per-date and summary similarity files.
    date_strs : Sequence[str]
        Date strings (``YYYYMMDD``) parallel to ``slc_file_list``.
    output_name : str
        Filename (not full path) of the aggregated similarity raster.
    search_radius : int
        Maximum radius (pixels) for spatial neighbor search.
    nearest_n : int
        Number of nearest temporal neighbors on each side to form
        interferograms for each date.
    sim_type : {"median", "max"}
        Spatial aggregation function.
    block_shape : tuple[int, int]
        Processing block size.
    num_threads : int
        Parallel processing threads.

    Returns
    -------
    tuple[list[Path], Path]
        ``(per_date_files, summary_file)`` where ``per_date_files`` are the
        individual ``similarity_{date}.tif`` outputs and ``summary_file`` is
        their pixel-wise mean.

    """
    from dolphin._overviews import Resampling, create_image_overviews
    from dolphin.io import load_gdal, write_arr

    output_folder = Path(output_folder)
    slc_file_list = [Path(f) for f in slc_file_list]
    n_slcs = len(slc_file_list)

    if sim_type == "median":
        sim_function = median_similarity
    elif sim_type == "max":
        sim_function = max_similarity
    else:
        raise ValueError(f"Unrecognized {sim_type = }")

    per_date_files: list[Path] = []

    for i, (slc_file, date_str) in enumerate(
        zip(slc_file_list, date_strs, strict=False)
    ):
        out_file = output_folder / f"similarity_{date_str}.tif"
        if out_file.exists():
            logger.info(f"{out_file.name} exists, skipping")
            per_date_files.append(out_file)
            continue

        # Gather nearest-N neighbors (clamped to stack bounds)
        neighbor_idxs = [
            j
            for j in range(max(0, i - nearest_n), min(n_slcs, i + nearest_n + 1))
            if j != i
        ]
        if not neighbor_idxs:
            logger.warning(f"No neighbors for {date_str}, skipping similarity")
            continue

        # Load the reference SLC and neighbors directly (avoids VRTStack sort-by-name)
        ref_slc = load_gdal(slc_file).astype(np.complex64)
        neighbor_slcs = np.stack(
            [load_gdal(slc_file_list[j]).astype(np.complex64) for j in neighbor_idxs],
            axis=0,
        )
        # Form interferograms: slc[i] * conj(slc[j]) for each neighbor
        ifg_stack = ref_slc[np.newaxis] * np.conj(neighbor_slcs)

        sim = sim_function(ifg_stack=ifg_stack, search_radius=search_radius)
        write_arr(
            arr=sim,
            like_filename=slc_file,
            output_name=out_file,
            nodata=np.nan,
        )
        logger.info(f"Written {out_file.name}")
        per_date_files.append(out_file)

    # Aggregate per-date files into a summary raster using median
    summary_file = output_folder / output_name
    if not summary_file.exists() and per_date_files:
        stack = np.stack(
            [load_gdal(f, masked=True).filled(np.nan) for f in per_date_files], axis=0
        )
        # Note nanmean is similar to the stack dolphin ph sim estimate
        #      nanmedian gives slightly higher values
        agg = np.nanmean(stack, axis=0).astype("float32")
        write_arr(
            arr=agg,
            like_filename=per_date_files[0],
            output_name=summary_file,
            nodata=np.nan,
        )
        create_image_overviews(summary_file, resampling=Resampling.AVERAGE)

    return per_date_files, summary_file


def create_per_date_similarities(
    ifg_file_list: Sequence[PathOrStr],
    output_dir: PathOrStr,
    search_radius: int = 7,
    sim_type: Literal["median", "max"] = "median",
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
    add_overviews: bool = True,
    nearest_n: int | None = None,
) -> list[Path]:
    """Create one similarity raster per SLC date from a stack of interferogram files.

    For each SLC date (index ``i`` in the stack), similarity is computed using
    only the interferograms that involve that date, giving a per-epoch quality map.

    Parameters
    ----------
    ifg_file_list : Sequence[PathOrStr]
        Paths to input interferograms, named ``{ref_date}_{sec_date}.tif``
        (or any filename containing exactly two 8-digit date strings).
    output_dir : PathOrStr
        Directory where per-date similarity rasters are written.
        Files are named ``similarity_{YYYYMMDD}.tif``.
    search_radius : int, optional
        Maximum radius to search for pixels, by default 7.
    sim_type : str, optional
        Similarity aggregation function: ``"median"`` or ``"max"``, by default
        ``"median"``.
    block_shape : tuple[int, int], optional
        Block size for processing, by default (512, 512).
    num_threads : int, optional
        Number of parallel blocks, by default 5.
    add_overviews : bool, optional
        Whether to add overviews to each output, by default True.
    nearest_n : int, optional
        If provided, reform the nearest-N interferograms before computing similarity.

    Returns
    -------
    list[Path]
        Sorted list of per-date output raster paths.

    """
    import re

    from dolphin._overviews import Resampling, create_image_overviews
    from dolphin.io import BackgroundRasterWriter, VRTStack, process_blocks

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = [Path(f) for f in ifg_file_list]

    # Parse (ref_date, sec_date) pairs from filenames
    date_pattern = re.compile(r"(\d{8})")
    pairs: list[tuple[str, str]] = []
    for f in paths:
        dates = date_pattern.findall(f.stem)
        if len(dates) < 2:
            raise ValueError(f"Could not parse two dates from filename: {f.name}")
        pairs.append((dates[0], dates[1]))

    # Collect all unique dates in order of first appearance
    all_dates: list[str] = []
    for ref, sec in pairs:
        if ref not in all_dates:
            all_dates.append(ref)
        if sec not in all_dates:
            all_dates.append(sec)

    if sim_type == "median":
        sim_function = median_similarity
    elif sim_type == "max":
        sim_function = max_similarity
    else:
        raise ValueError(f"Unrecognized {sim_type = }")

    nodata_block = np.full(block_shape, fill_value=np.nan, dtype="float32")

    if nearest_n is not None:
        from dolphin.timeseries import get_incidence_matrix

        incidence_matrix = get_incidence_matrix(
            _create_nearest_n_pairs(len(ifg_file_list) + 1, n=nearest_n)
        )
        assert incidence_matrix.shape[1] == len(ifg_file_list)
    else:
        incidence_matrix = None

    output_files: list[Path] = []

    for date in all_dates:
        output_file = output_dir / f"similarity_{date}.tif"
        if output_file.exists():
            logger.info(f"{output_file} exists, skipping")
            output_files.append(output_file)
            continue

        # Select interferograms that involve this date
        date_idxs = [
            i for i, (ref, sec) in enumerate(pairs) if ref == date or sec == date
        ]
        if not date_idxs:
            continue

        date_files = [ifg_file_list[i] for i in date_idxs]
        reader = VRTStack(date_files, outfile=output_dir / f"sim_inputs_{date}.vrt")

        if incidence_matrix is not None:
            sub_incidence = incidence_matrix[:, date_idxs]
        else:
            sub_incidence = None

        def calc_sim(readers, rows, cols, _sub_inc=sub_incidence):
            block = readers[0][:, rows, cols]
            if np.sum(block) == 0 or np.isnan(block).all():
                return nodata_block[rows, cols], rows, cols
            if _sub_inc is not None:
                block = _calc_nearest_diffs(block, _sub_inc)
            out_avg = sim_function(ifg_stack=block, search_radius=search_radius)
            return out_avg, rows, cols

        writer = BackgroundRasterWriter(
            output_file,
            like_filename=date_files[0],
            dtype="float32",
            driver="GTiff",
            nodata=np.nan,
        )
        process_blocks(
            [reader],
            writer,
            func=calc_sim,
            block_shape=block_shape,
            overlaps=(search_radius, search_radius),
            num_threads=num_threads,
        )
        writer.notify_finished()

        if add_overviews:
            create_image_overviews(output_file, resampling=Resampling.AVERAGE)

        output_files.append(output_file)

    return sorted(output_files)


def _calc_nearest_diffs(block, incidence_matrix) -> np.ndarray:
    # Multiply the single-ref data by tall and skinny A matrix
    # to give the nearest-n differences
    num_imgs, rows, cols = block.shape
    block_mask = np.nan_to_num(block).sum(axis=0) == 0
    m, num_imgs = incidence_matrix.shape
    phase = np.angle(block) if np.iscomplexobj(block) else block
    columns = np.dot(incidence_matrix, phase.reshape(num_imgs, -1))
    block = columns.reshape(m, rows, cols)
    block[:, block_mask] = np.nan
    return np.exp(1j * block)


def _create_nearest_n_pairs(num_files: int, n: int = 3) -> list[tuple[int, int]]:
    """Create nearest-n interferogram pair indices for a list of `num_files` inputs."""
    ijs = []
    for i in range(num_files):
        for j in range(i + 1, i + n + 1):
            if j >= num_files:
                continue
            ijs.append((i, j))
    return ijs
