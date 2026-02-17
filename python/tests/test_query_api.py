"""Tests for TraceData query API methods."""

import pytest

from prlx.trace_reader import read_trace


@pytest.fixture
def sample_trace(trace_builder):
    """A trace with mixed event types across 2 warps."""
    return trace_builder(
        kernel_name="query_test",
        grid_dim=(1, 1, 1),
        block_dim=(64, 1, 1),
        warp_events=[
            [
                # warp 0: 2 branches, 1 atomic
                {"site_id": 0xA000, "event_type": 0, "branch_dir": 1, "active_mask": 0xFFFFFFFF, "value_a": 10},
                {"site_id": 0xA001, "event_type": 0, "branch_dir": 0, "active_mask": 0xFFFF0000, "value_a": 20},
                {"site_id": 0xB000, "event_type": 2, "branch_dir": 0, "active_mask": 0xFFFFFFFF, "value_a": 99},
            ],
            [
                # warp 1: 1 branch, 1 shmem store
                {"site_id": 0xA000, "event_type": 0, "branch_dir": 1, "active_mask": 0xFFFFFFFF, "value_a": 10},
                {"site_id": 0xC000, "event_type": 1, "branch_dir": 0, "active_mask": 0xFFFFFFFF, "value_a": 42},
            ],
        ],
    )


def test_filter_events_by_type(sample_trace):
    td = read_trace(str(sample_trace))
    branches = td.filter_events(event_type=0)
    assert len(branches) == 3  # 2 from warp 0, 1 from warp 1


def test_filter_events_by_site(sample_trace):
    td = read_trace(str(sample_trace))
    events = td.filter_events(site_id=0xA000)
    assert len(events) == 2  # one in each warp


def test_filter_events_by_warp(sample_trace):
    td = read_trace(str(sample_trace))
    events = td.filter_events(warp_idx=0)
    assert len(events) == 3


def test_filter_events_combined(sample_trace):
    td = read_trace(str(sample_trace))
    events = td.filter_events(event_type=0, warp_idx=0)
    assert len(events) == 2


def test_events_at_site(sample_trace):
    td = read_trace(str(sample_trace))
    pairs = td.events_at_site(0xA000)
    assert len(pairs) == 2
    warp_indices = [p[0] for p in pairs]
    assert 0 in warp_indices
    assert 1 in warp_indices


def test_branches_shorthand(sample_trace):
    td = read_trace(str(sample_trace))
    assert len(td.branches()) == 3


def test_atomics_shorthand(sample_trace):
    td = read_trace(str(sample_trace))
    assert len(td.atomics()) == 1
    assert td.atomics()[0].site_id == 0xB000


def test_summary(sample_trace):
    td = read_trace(str(sample_trace))
    s = td.summary()
    assert s["kernel_name"] == "query_test"
    assert s["num_warps"] == 2
    assert s["total_events"] == 5
    assert s["events_by_type"]["Branch"] == 3
    assert s["events_by_type"]["Atomic"] == 1
    assert s["events_by_type"]["SharedMemStore"] == 1


def test_compare_warps_identical_site(sample_trace):
    td = read_trace(str(sample_trace))
    diffs = td.compare_warps(0, 1)
    # Index 0: both have site_id 0xA000 with same branch_dir → no diff at index 0
    # Index 1: different site_id → diff
    # Index 2: warp 0 has event, warp 1 doesn't → missing
    assert len(diffs) >= 2
    # At least one diff should have "missing" in fields (warp 1 shorter)
    missing_diffs = [d for d in diffs if "missing" in d["fields"]]
    assert len(missing_diffs) >= 1


def test_compare_warps_branch_dir_diff(trace_builder):
    path = trace_builder(
        warp_events=[
            [{"site_id": 0x1000, "event_type": 0, "branch_dir": 1}],
            [{"site_id": 0x1000, "event_type": 0, "branch_dir": 0}],
        ],
    )
    td = read_trace(str(path))
    diffs = td.compare_warps(0, 1)
    assert len(diffs) == 1
    assert "branch_dir" in diffs[0]["fields"]
