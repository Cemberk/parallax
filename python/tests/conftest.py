"""Trace file builder fixtures for PRLX tests."""

import struct
import tempfile
from pathlib import Path

import pytest

PRLX_MAGIC = 0x50524C5800000000
PRLX_VERSION = 1
PRLX_FLAG_COMPRESS = 0x2
PRLX_FLAG_HISTORY = 0x4
PRLX_FLAG_SAMPLED = 0x8
PRLX_FLAG_SNAPSHOT = 0x10

HEADER_SIZE = 160
TRACE_EVENT_SIZE = 16
WARP_BUFFER_HEADER_SIZE = 16
HISTORY_RING_HEADER_SIZE = 16
HISTORY_ENTRY_SIZE = 16
SNAPSHOT_RING_HEADER_SIZE = 16
SNAPSHOT_ENTRY_SIZE = 288

_HEADER_FMT = "<QII Q 64s 3I 3I II I 4x Q I II 3I"
_WARP_HEADER_FMT = "<IIII"
_EVENT_FMT = "<I BB H I I"
_HISTORY_RING_FMT = "<IIII"
_HISTORY_ENTRY_FMT = "<IIII"
_SNAPSHOT_RING_FMT = "<IIII"
_SNAPSHOT_ENTRY_FMT = "<IIII 32I 32I 4I"

DEFAULT_EVENTS_PER_WARP = 128  # Small for tests (real default is 4096)


def make_event(site_id=0, event_type=0, branch_dir=0, active_mask=0xFFFFFFFF, value_a=0):
    return struct.pack(_EVENT_FMT, site_id, event_type, branch_dir, 0, active_mask, value_a)


def make_trace(
    kernel_name="test_kernel",
    kernel_hash=0x1234567890ABCDEF,
    grid_dim=(1, 1, 1),
    block_dim=(32, 1, 1),
    warp_events=None,
    warp_overflows=None,
    events_per_warp=DEFAULT_EVENTS_PER_WARP,
    flags=0,
    cuda_arch=80,
    timestamp=1234567890,
    sample_rate=1,
    history_depth=0,
    history_data=None,
    snapshot_depth=0,
    snapshot_data=None,
):
    if warp_events is None:
        warp_events = [[]]

    num_warps = len(warp_events)
    if warp_overflows is None:
        warp_overflows = [0] * num_warps

    # Calculate section offsets
    warp_buffer_size = WARP_BUFFER_HEADER_SIZE + events_per_warp * TRACE_EVENT_SIZE
    event_section_end = HEADER_SIZE + num_warps * warp_buffer_size

    history_section_offset = 0
    if history_depth > 0 and history_data:
        flags |= PRLX_FLAG_HISTORY
        history_section_offset = event_section_end

    snapshot_section_offset = 0
    if snapshot_depth > 0 and snapshot_data:
        flags |= PRLX_FLAG_SNAPSHOT
        hist_size = 0
        if history_section_offset > 0:
            hist_ring_size = HISTORY_RING_HEADER_SIZE + history_depth * HISTORY_ENTRY_SIZE
            hist_size = num_warps * hist_ring_size
        snapshot_section_offset = event_section_end + hist_size

    # Encode kernel name (64 bytes, null-padded)
    name_bytes = kernel_name.encode("utf-8")[:63]
    name_padded = name_bytes + b"\x00" * (64 - len(name_bytes))

    # Pack header
    header = struct.pack(
        _HEADER_FMT,
        PRLX_MAGIC,
        PRLX_VERSION,
        flags,
        kernel_hash,
        name_padded,
        grid_dim[0], grid_dim[1], grid_dim[2],
        block_dim[0], block_dim[1], block_dim[2],
        (block_dim[0] * block_dim[1] * block_dim[2] + 31) // 32,  # warps_per_block
        num_warps,
        events_per_warp,
        timestamp,
        cuda_arch,
        history_depth,
        history_section_offset,
        sample_rate,
        snapshot_depth,
        snapshot_section_offset,
    )
    assert len(header) == HEADER_SIZE

    # Write to temp file
    tf = tempfile.NamedTemporaryFile(suffix=".prlx", delete=False)

    tf.write(header)

    # Write warp buffers
    for w_idx, events in enumerate(warp_events):
        num_events = len(events)
        overflow = warp_overflows[w_idx] if w_idx < len(warp_overflows) else 0
        warp_hdr = struct.pack(_WARP_HEADER_FMT, num_events, overflow, num_events, 0)
        tf.write(warp_hdr)

        for evt in events:
            if isinstance(evt, tuple):
                tf.write(make_event(*evt))
            elif isinstance(evt, bytes):
                tf.write(evt)
            else:
                tf.write(make_event(**evt))

        # Pad remaining slots
        padding = (events_per_warp - num_events) * TRACE_EVENT_SIZE
        tf.write(b"\x00" * padding)

    # Write history section
    if history_depth > 0 and history_data:
        for w_idx in range(num_warps):
            entries = history_data[w_idx] if w_idx < len(history_data) else []
            ring_hdr = struct.pack(_HISTORY_RING_FMT, len(entries), history_depth, len(entries), 0)
            tf.write(ring_hdr)

            for site_id, value, seq in entries:
                tf.write(struct.pack(_HISTORY_ENTRY_FMT, site_id, value, seq, 0))

            # Pad remaining slots
            remaining = history_depth - len(entries)
            tf.write(b"\x00" * (remaining * HISTORY_ENTRY_SIZE))

    # Write snapshot section
    if snapshot_depth > 0 and snapshot_data:
        for w_idx in range(num_warps):
            snaps = snapshot_data[w_idx] if w_idx < len(snapshot_data) else []
            ring_hdr = struct.pack(_SNAPSHOT_RING_FMT, len(snaps), snapshot_depth, len(snaps), 0)
            tf.write(ring_hdr)

            for snap in snaps:
                lhs = snap.get("lhs_values", [0] * 32)
                rhs = snap.get("rhs_values", [0] * 32)
                pad = [0, 0, 0, 0]
                tf.write(struct.pack(
                    _SNAPSHOT_ENTRY_FMT,
                    snap["site_id"], snap.get("active_mask", 0xFFFFFFFF),
                    snap.get("seq", 0), snap.get("cmp_predicate", 0),
                    *lhs, *rhs, *pad,
                ))

            remaining = snapshot_depth - len(snaps)
            tf.write(b"\x00" * (remaining * SNAPSHOT_ENTRY_SIZE))

    tf.flush()
    tf.close()
    return Path(tf.name)


@pytest.fixture
def trace_builder():
    created_files = []

    def _build(**kwargs):
        path = make_trace(**kwargs)
        created_files.append(path)
        return path

    yield _build

    for f in created_files:
        try:
            f.unlink()
        except OSError:
            pass
