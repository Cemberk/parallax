"""
Pure-Python trace file reader using struct and mmap.

Reads the binary .prlx trace format without requiring the Rust differ.
Supports the full format including history ring buffers (Phase 8).
"""

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# Constants (must match trace_format.h)
PRLX_MAGIC = 0x50524C5800000000
PRLX_VERSION = 1
PRLX_FLAG_COMPRESS = 0x2
PRLX_FLAG_HISTORY = 0x4
PRLX_FLAG_SAMPLED = 0x8
PRLX_FLAG_SNAPSHOT = 0x10

# Struct sizes (must match C layout)
HEADER_SIZE = 160
TRACE_EVENT_SIZE = 16
WARP_BUFFER_HEADER_SIZE = 16
HISTORY_RING_HEADER_SIZE = 16
HISTORY_ENTRY_SIZE = 16
SNAPSHOT_RING_HEADER_SIZE = 16
SNAPSHOT_ENTRY_SIZE = 288

# struct formats (little-endian)
# Note: 4x padding before timestamp for uint64_t 8-byte alignment (offset 124→128)
_HEADER_FMT = "<QII Q 64s 3I 3I II I 4x Q I II 3I"
_WARP_HEADER_FMT = "<IIII"
_EVENT_FMT = "<I BB H I I"
_HISTORY_RING_FMT = "<IIII"
_HISTORY_ENTRY_FMT = "<IIII"
_SNAPSHOT_RING_FMT = "<IIII"
_SNAPSHOT_ENTRY_FMT = "<IIII 32I 32I 4I"  # site_id, mask, seq, pred, lhs[32], rhs[32], pad[4]

# Event type names
EVENT_NAMES = {
    0: "Branch",
    1: "SharedMemStore",
    2: "Atomic",
    3: "FuncEntry",
    4: "FuncExit",
    5: "Switch",
}


@dataclass
class TraceEvent:
    """A single trace event."""
    site_id: int
    event_type: int
    branch_dir: int
    active_mask: int
    value_a: int

    @property
    def event_type_name(self) -> str:
        return EVENT_NAMES.get(self.event_type, f"Unknown({self.event_type})")

    @property
    def is_branch(self) -> bool:
        return self.event_type == 0

    @property
    def branch_taken(self) -> bool:
        return self.branch_dir != 0

    @property
    def active_thread_count(self) -> int:
        return bin(self.active_mask).count("1")


@dataclass
class HistoryEntry:
    """A single history ring buffer entry."""
    site_id: int
    value: int
    seq: int


@dataclass
class SnapshotEntry:
    """A single per-lane comparison operand snapshot."""
    site_id: int
    active_mask: int
    seq: int
    cmp_predicate: int
    lhs_values: List[int]  # 32 per-lane LHS operands
    rhs_values: List[int]  # 32 per-lane RHS operands


@dataclass
class WarpData:
    """Events and metadata for a single warp."""
    warp_idx: int
    num_events: int
    overflow_count: int
    events: List[TraceEvent]
    history: List[HistoryEntry] = field(default_factory=list)
    snapshots: List[SnapshotEntry] = field(default_factory=list)


@dataclass
class TraceHeader:
    """Parsed trace file header."""
    magic: int
    version: int
    flags: int
    kernel_name_hash: int
    kernel_name: str
    grid_dim: tuple
    block_dim: tuple
    num_warps_per_block: int
    total_warp_slots: int
    events_per_warp: int
    timestamp: int
    cuda_arch: int
    history_depth: int
    history_section_offset: int
    sample_rate: int = 1
    snapshot_depth: int = 0
    snapshot_section_offset: int = 0

    @property
    def has_history(self) -> bool:
        return bool(self.flags & PRLX_FLAG_HISTORY) and self.history_depth > 0

    @property
    def has_snapshot(self) -> bool:
        return bool(self.flags & PRLX_FLAG_SNAPSHOT) and self.snapshot_depth > 0

    @property
    def is_sampled(self) -> bool:
        return bool(self.flags & PRLX_FLAG_SAMPLED) and self.sample_rate > 1

    @property
    def total_blocks(self) -> int:
        return self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

    @property
    def threads_per_block(self) -> int:
        return self.block_dim[0] * self.block_dim[1] * self.block_dim[2]


class TraceData:
    """
    Parsed trace file providing access to header, warps, and history.

    Usage:
        trace = read_trace("trace.prlx")
        print(trace.header.kernel_name)
        for warp in trace.warps():
            for event in warp.events:
                print(event)
    """

    def __init__(self, header: TraceHeader, warp_data: List[WarpData]):
        self._header = header
        self._warps = warp_data

    @property
    def header(self) -> TraceHeader:
        return self._header

    def warps(self) -> Sequence[WarpData]:
        """Iterate over all warps."""
        return self._warps

    def warp(self, idx: int) -> WarpData:
        """Get data for a specific warp."""
        if idx < 0 or idx >= len(self._warps):
            raise IndexError(f"Warp index {idx} out of range (0..{len(self._warps)})")
        return self._warps[idx]

    @property
    def num_warps(self) -> int:
        return len(self._warps)

    @property
    def total_events(self) -> int:
        return sum(w.num_events for w in self._warps)

    @property
    def total_overflows(self) -> int:
        return sum(w.overflow_count for w in self._warps)

    def divergent_warps(self, other: "TraceData") -> List[int]:
        """Quick check: which warps have different event counts?"""
        result = []
        for i in range(min(self.num_warps, other.num_warps)):
            if self._warps[i].num_events != other._warps[i].num_events:
                result.append(i)
        return result


def read_trace(path: str) -> TraceData:
    """
    Read a .prlx trace file (handles zstd-compressed files transparently).

    Args:
        path: Path to the trace file.

    Returns:
        TraceData object with parsed header, warps, and history.

    Raises:
        ValueError: If the file has invalid magic or version.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # Check for compression: parse flags from header
            if len(mm) >= HEADER_SIZE:
                flags = struct.unpack_from("<I", mm, 12)[0]  # flags at offset 12
                if flags & PRLX_FLAG_COMPRESS:
                    return _read_compressed(mm)
            return _parse_trace(mm)
        finally:
            mm.close()


def _read_compressed(mm) -> TraceData:
    """Decompress a zstd-compressed trace and parse it."""
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard package required for compressed traces. "
            "Install with: pip install zstandard"
        )

    # Header is uncompressed (first 160 bytes)
    header_bytes = bytes(mm[:HEADER_SIZE])
    compressed_payload = bytes(mm[HEADER_SIZE:])

    # Decompress payload
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed_payload)

    # Combine header + decompressed payload
    full_data = header_bytes + decompressed
    return _parse_trace(full_data)


def _parse_header(data: bytes) -> TraceHeader:
    """Parse the 160-byte file header."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small for header: {len(data)} bytes")

    # Unpack fields
    values = struct.unpack_from(_HEADER_FMT, data, 0)

    magic = values[0]
    version = values[1]
    flags = values[2]
    kernel_name_hash = values[3]
    kernel_name_raw = values[4]
    gx, gy, gz = values[5], values[6], values[7]
    bx, by, bz = values[8], values[9], values[10]
    num_warps_per_block = values[11]
    total_warp_slots = values[12]
    events_per_warp = values[13]
    timestamp = values[14]
    cuda_arch = values[15]
    history_depth = values[16]
    history_section_offset = values[17]
    sample_rate = values[18]  # 0 or 1 = record all, N = record 1/N
    snapshot_depth = values[19]  # 0 = no snapshots
    snapshot_section_offset = values[20]  # 0 = auto (after history)

    # Validate
    if magic != PRLX_MAGIC:
        raise ValueError(
            f"Invalid magic: 0x{magic:016x} (expected 0x{PRLX_MAGIC:016x})"
        )
    if version != PRLX_VERSION:
        raise ValueError(f"Unsupported version: {version} (expected {PRLX_VERSION})")

    # Decode kernel name (null-terminated)
    kernel_name = kernel_name_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    return TraceHeader(
        magic=magic,
        version=version,
        flags=flags,
        kernel_name_hash=kernel_name_hash,
        kernel_name=kernel_name,
        grid_dim=(gx, gy, gz),
        block_dim=(bx, by, bz),
        num_warps_per_block=num_warps_per_block,
        total_warp_slots=total_warp_slots,
        events_per_warp=events_per_warp,
        timestamp=timestamp,
        cuda_arch=cuda_arch,
        history_depth=history_depth,
        history_section_offset=history_section_offset,
        sample_rate=max(sample_rate, 1),
        snapshot_depth=snapshot_depth,
        snapshot_section_offset=snapshot_section_offset,
    )


def _parse_trace(mm: mmap.mmap) -> TraceData:
    """Parse the full trace file from a memory-mapped buffer."""
    header = _parse_header(mm[:HEADER_SIZE])

    warp_buffer_size = WARP_BUFFER_HEADER_SIZE + header.events_per_warp * TRACE_EVENT_SIZE

    warps = []
    for w in range(header.total_warp_slots):
        offset = HEADER_SIZE + w * warp_buffer_size

        # Parse warp header
        wh = struct.unpack_from(_WARP_HEADER_FMT, mm, offset)
        num_events = min(wh[2], header.events_per_warp)  # num_events field
        overflow_count = wh[1]

        # Parse events
        events = []
        evt_offset = offset + WARP_BUFFER_HEADER_SIZE
        for e in range(num_events):
            ev = struct.unpack_from(_EVENT_FMT, mm, evt_offset + e * TRACE_EVENT_SIZE)
            events.append(TraceEvent(
                site_id=ev[0],
                event_type=ev[1],
                branch_dir=ev[2],
                active_mask=ev[4],
                value_a=ev[5],
            ))

        warps.append(WarpData(
            warp_idx=w,
            num_events=num_events,
            overflow_count=overflow_count,
            events=events,
        ))

    # Parse history section if present
    if header.has_history:
        history_offset = (
            header.history_section_offset
            if header.history_section_offset > 0
            else HEADER_SIZE + header.total_warp_slots * warp_buffer_size
        )
        ring_size = HISTORY_RING_HEADER_SIZE + header.history_depth * HISTORY_ENTRY_SIZE

        for w in range(header.total_warp_slots):
            ring_offset = history_offset + w * ring_size

            # Check bounds
            if ring_offset + ring_size > len(mm):
                break

            # Parse ring header
            rh = struct.unpack_from(_HISTORY_RING_FMT, mm, ring_offset)
            depth = rh[1]
            total_writes = rh[2]

            if total_writes == 0:
                continue

            # Parse entries
            valid_count = min(total_writes, depth)
            entries = []
            ent_offset = ring_offset + HISTORY_RING_HEADER_SIZE
            for e in range(min(depth, header.history_depth)):
                he = struct.unpack_from(
                    _HISTORY_ENTRY_FMT, mm, ent_offset + e * HISTORY_ENTRY_SIZE
                )
                if total_writes <= depth and e >= total_writes:
                    break  # Ring hasn't wrapped, skip unwritten slots
                entries.append(HistoryEntry(site_id=he[0], value=he[1], seq=he[2]))

            # Sort by sequence number for chronological order
            entries.sort(key=lambda e: e.seq)
            warps[w].history = entries

    # Parse snapshot section if present
    if header.has_snapshot:
        # Compute snapshot section offset
        if header.snapshot_section_offset > 0:
            snap_offset = header.snapshot_section_offset
        else:
            # Auto: after history (or after event buffers if no history)
            base = HEADER_SIZE + header.total_warp_slots * warp_buffer_size
            if header.has_history:
                hist_ring_size = HISTORY_RING_HEADER_SIZE + header.history_depth * HISTORY_ENTRY_SIZE
                base += header.total_warp_slots * hist_ring_size
            snap_offset = base

        snap_ring_size = SNAPSHOT_RING_HEADER_SIZE + header.snapshot_depth * SNAPSHOT_ENTRY_SIZE

        for w in range(header.total_warp_slots):
            ring_offset = snap_offset + w * snap_ring_size

            # Check bounds
            if ring_offset + snap_ring_size > len(mm):
                break

            # Parse ring header
            rh = struct.unpack_from(_SNAPSHOT_RING_FMT, mm, ring_offset)
            depth = rh[1]
            total_writes = rh[2]

            if total_writes == 0:
                continue

            # Parse entries
            entries = []
            ent_offset = ring_offset + SNAPSHOT_RING_HEADER_SIZE
            for e in range(min(depth, header.snapshot_depth)):
                if total_writes <= depth and e >= total_writes:
                    break  # Ring hasn't wrapped
                se = struct.unpack_from(
                    _SNAPSHOT_ENTRY_FMT, mm, ent_offset + e * SNAPSHOT_ENTRY_SIZE
                )
                # se layout: site_id, mask, seq, pred, lhs[32], rhs[32], pad[4]
                entries.append(SnapshotEntry(
                    site_id=se[0],
                    active_mask=se[1],
                    seq=se[2],
                    cmp_predicate=se[3],
                    lhs_values=list(se[4:36]),
                    rhs_values=list(se[36:68]),
                ))

            # Sort by sequence number
            entries.sort(key=lambda e: e.seq)
            warps[w].snapshots = entries

    return TraceData(header, warps)
