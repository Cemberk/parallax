"""Tests for prlx.viz visualization helpers."""

import pytest

from prlx.trace_reader import read_trace
from prlx.viz import display_trace_summary


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def sample_trace(trace_builder):
    return trace_builder(
        kernel_name="viz_test",
        warp_events=[
            [
                {"site_id": 0x1000, "event_type": 0, "branch_dir": 1, "active_mask": 0xFFFFFFFF},
                {"site_id": 0x2000, "event_type": 2, "branch_dir": 0, "active_mask": 0xFFFF0000},
            ],
        ],
    )


def test_display_trace_summary_returns_html(sample_trace):
    td = read_trace(str(sample_trace))
    html = display_trace_summary(td)
    assert "<table" in html
    assert "viz_test" in html
    assert "Branch" in html
    assert "Atomic" in html


def test_display_trace_summary_no_matplotlib(sample_trace):
    """display_trace_summary should work without matplotlib."""
    td = read_trace(str(sample_trace))
    html = display_trace_summary(td)
    assert isinstance(html, str)
    assert len(html) > 50


@pytest.mark.skipif(
    not _matplotlib_available(),
    reason="matplotlib not installed",
)
def test_plot_warp_timeline(sample_trace):
    from prlx.viz import plot_warp_timeline
    td = read_trace(str(sample_trace))
    fig = plot_warp_timeline(td)
    assert fig is not None


@pytest.mark.skipif(
    not _matplotlib_available(),
    reason="matplotlib not installed",
)
def test_plot_divergence_heatmap(sample_trace, trace_builder):
    from prlx.viz import plot_divergence_heatmap
    td_a = read_trace(str(sample_trace))
    other = trace_builder(
        kernel_name="viz_test",
        warp_events=[
            [
                {"site_id": 0x1000, "event_type": 0, "branch_dir": 0, "active_mask": 0xFFFFFFFF},
            ],
        ],
    )
    td_b = read_trace(str(other))
    fig = plot_divergence_heatmap(td_a, td_b)
    assert fig is not None
