"""Manual-index networks and the compressed SLC's reference epoch.

A manual-index network addresses the phase-linked (real-date) list only. When
the compressed SLC's reference epoch falls inside the window the indexes span,
the interval from it to the next date has no interferogram unless the
(reference -> real) ifgs are kept.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from dolphin.workflows.config import InterferogramNetwork
from dolphin.workflows.wrapped_phase import compressed_reference_ifgs, create_ifgs

NEAREST_4 = [(-2, -1), (-3, -1), (-4, -1), (-3, -2), (-4, -2), (-4, -3),
             (-5, -1), (-5, -2), (-5, -3), (-5, -4)]
D0 = datetime(2019, 8, 13)


def _dates(n, skip=()):
    out = [D0 + timedelta(days=6 * k) for k in range(n)]
    return [d for k, d in enumerate(out) if k not in skip]


def _ifgs(ref, dates):
    return [Path(f"{ref:%Y%m%d}_{d:%Y%m%d}.int.vrt") for d in dates]


def test_nothing_added_when_the_reference_predates_the_window():
    """The normal forward run: compressed epoch well before the last 5 dates."""
    dates = _dates(14)
    ref = D0 - timedelta(days=6)
    assert compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates)) == []


def test_the_missing_last_interval_is_added():
    """The run after a compression: its reference is the second-to-last date,
    and its real image is absent (a real SLC may not share the date)."""
    all_dates = _dates(15)
    ref = all_dates[-2]
    dates = [d for d in all_dates if d != ref]
    got = compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates))
    assert got == [Path(f"{ref:%Y%m%d}_{all_dates[-1]:%Y%m%d}.int.vrt")]


def test_only_dates_after_the_reference_inside_the_window():
    all_dates = _dates(15)
    ref = all_dates[-4]
    dates = [d for d in all_dates if d != ref]
    got = compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates))
    assert got == _ifgs(ref, all_dates[-3:])


def test_positive_indexes_are_not_guessed_at():
    dates = _dates(14)
    ref = dates[-2]
    assert compressed_reference_ifgs([(0, 1)], ref, dates, _ifgs(ref, dates)) == []


def test_datetime_and_date_compare_as_dates():
    """reference_date carries a time of day; filename dates do not."""
    all_dates = _dates(15)
    ref = all_dates[-2] + timedelta(hours=14, minutes=8)
    dates = [d for d in all_dates if d != all_dates[-2]]
    assert len(compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates))) == 1


def _phase_linked(tmp_path, dates):
    out = []
    for d in dates:
        p = tmp_path / f"{d:%Y%m%d}.slc.tif"
        p.touch()
        out.append(p)
    return out


@pytest.mark.parametrize("include", [False, True])
def test_create_ifgs_keeps_the_reference_ifg_only_when_asked(tmp_path, include):
    all_dates = _dates(15)
    ref = all_dates[-2]
    pl = _phase_linked(tmp_path, [d for d in all_dates if d != ref])
    net = InterferogramNetwork(indexes=NEAREST_4, include_compressed_reference=include)
    net._directory = tmp_path / "ifgs"
    names = {p.name for p in create_ifgs(net, pl, True, ref, dry_run=True)}
    wanted = f"{ref:%Y%m%d}_{all_dates[-1]:%Y%m%d}"
    assert any(n.startswith(wanted) for n in names) is include
    assert len(names) == 10 + include, "the 10 nearest-4 real pairs always stay"


def test_the_default_leaves_existing_networks_alone():
    assert InterferogramNetwork(indexes=NEAREST_4).include_compressed_reference is False
