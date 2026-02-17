"""Pure-Python .prlx trace file reader using struct + mmap."""

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# Must match trace_format.h
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

# 4x padding before timestamp for uint64_t alignment (offset 124->128)
_HEADER_FMT = "<QII Q 64s 3I 3I II I 4x Q I II 3I"
_WARP_HEADER_FMT = "<IIII"
_EVENT_FMT = "<I BB H I I"
_HISTORY_RING_FMT = "<IIII"
_HISTORY_ENTRY_FMT = "<IIII"
_SNAPSHOT_RING_FMT = "<IIII"
_SNAPSHOT_ENTRY_FMT = "<IIII 32I 32I 4I"

EVENT_NAMES = {
    0: "Branch",
    1: "SharedMemStore",
    2: "Atomic",
    3: "FuncEntry",
    4: "FuncExit",
    5: "Switch",
    6: "GlobalStore",
}


@dataclass
class TraceEvent:
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
    site_id: int
    value: int
    seq: int


@dataclass
class SnapshotEntry:
    site_id: int
    active_mask: int
    seq: int
    cmp_predicate: int
    lhs_values: List[int]
    rhs_values: List[int]


@dataclass
class WarpData:
    warp_idx: int
    num_events: int
    overflow_count: int
    events: List[TraceEvent]
    history: List[HistoryEntry] = field(default_factory=list)
    snapshots: List[SnapshotEntry] = field(default_factory=list)


@dataclass
class TraceHeader:
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

    def __init__(self, header: TraceHeader, warp_data: List[WarpData]):
        self._header = header
        self._warps = warp_data

    @property
    def header(self) -> TraceHeader:
        return self._header

    def warps(self) -> Sequence[WarpData]:
        return self._warps

    def warp(self, idx: int) -> WarpData:
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
        """Which warps have different event counts?"""
        result = []
        for i in range(min(self.num_warps, other.num_warps)):
            if self._warps[i].num_events != other._warps[i].num_events:
                result.append(i)
        return result

    def filter_events(
        self,
        event_type: Optional[int] = None,
        site_id: Optional[int] = None,
        warp_idx: Optional[int] = None,
    ) -> List[TraceEvent]:
        """Return events matching all specified criteria."""
        results: List[TraceEvent] = []
        for w in self._warps:
            if warp_idx is not None and w.warp_idx != warp_idx:
                continue
            for ev in w.events:
                if event_type is not None and ev.event_type != event_type:
                    continue
                if site_id is not None and ev.site_id != site_id:
                    continue
                results.append(ev)
        return results

    def events_at_site(self, site_id: int) -> List[tuple]:
        """Return ``(warp_idx, event)`` pairs for every event at *site_id*."""
        out: List[tuple] = []
        for w in self._warps:
            for ev in w.events:
                if ev.site_id == site_id:
                    out.append((w.warp_idx, ev))
        return out

    def branches(self) -> List[TraceEvent]:
        """Shorthand for ``filter_events(event_type=0)``."""
        return self.filter_events(event_type=0)

    def atomics(self) -> List[TraceEvent]:
        """Shorthand for ``filter_events(event_type=2)``."""
        return self.filter_events(event_type=2)

    def summary(self) -> dict:
        """Return a summary dict: kernel name, dims, event counts, overflows."""
        h = self._header
        counts: dict = {}
        for w in self._warps:
            for ev in w.events:
                name = EVENT_NAMES.get(ev.event_type, f"Unknown({ev.event_type})")
                counts[name] = counts.get(name, 0) + 1
        return {
            "kernel_name": h.kernel_name,
            "grid_dim": h.grid_dim,
            "block_dim": h.block_dim,
            "num_warps": self.num_warps,
            "total_events": self.total_events,
            "total_overflows": self.total_overflows,
            "events_by_type": counts,
        }

    def compare_warps(self, warp_a_idx: int, warp_b_idx: int) -> List[dict]:
        """Compare events of two warps, returning per-index field diffs."""
        wa = self.warp(warp_a_idx)
        wb = self.warp(warp_b_idx)
        diffs: List[dict] = []
        for i in range(max(len(wa.events), len(wb.events))):
            ea = wa.events[i] if i < len(wa.events) else None
            eb = wb.events[i] if i < len(wb.events) else None
            if ea is None or eb is None:
                diffs.append({"index": i, "a": ea, "b": eb, "fields": ["missing"]})
                continue
            changed = []
            if ea.site_id != eb.site_id:
                changed.append("site_id")
            if ea.event_type != eb.event_type:
                changed.append("event_type")
            if ea.branch_dir != eb.branch_dir:
                changed.append("branch_dir")
            if ea.active_mask != eb.active_mask:
                changed.append("active_mask")
            if ea.value_a != eb.value_a:
                changed.append("value_a")
            if changed:
                diffs.append({"index": i, "a": ea, "b": eb, "fields": changed})
        return diffs


def read_trace(path: str) -> TraceData:
    """Read a .prlx trace file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mm) >= HEADER_SIZE:
                flags = struct.unpack_from("<I", mm, 12)[0]
                if flags & PRLX_FLAG_COMPRESS:
                    return _read_compressed(mm)
            return _parse_trace(mm)
        finally:
            mm.close()


def _read_compressed(mm) -> TraceData:
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard package required for compressed traces. "
            "Install with: pip install zstandard"
        )

    header_bytes = bytes(mm[:HEADER_SIZE])
    compressed_payload = bytes(mm[HEADER_SIZE:])

    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(compressed_payload)

    return _parse_trace(header_bytes + decompressed)


def _parse_header(data: bytes) -> TraceHeader:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small for header: {len(data)} bytes")

    values = struct.unpack_from(_HEADER_FMT, data, 0)

    magic = values[0]
    if magic != PRLX_MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:016x} (expected 0x{PRLX_MAGIC:016x})")
    if values[1] != PRLX_VERSION:
        raise ValueError(f"Unsupported version: {values[1]} (expected {PRLX_VERSION})")

    kernel_name = values[4].split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    return TraceHeader(
        magic=magic,
        version=values[1],
        flags=values[2],
        kernel_name_hash=values[3],
        kernel_name=kernel_name,
        grid_dim=(values[5], values[6], values[7]),
        block_dim=(values[8], values[9], values[10]),
        num_warps_per_block=values[11],
        total_warp_slots=values[12],
        events_per_warp=values[13],
        timestamp=values[14],
        cuda_arch=values[15],
        history_depth=values[16],
        history_section_offset=values[17],
        sample_rate=max(values[18], 1),
        snapshot_depth=values[19],
        snapshot_section_offset=values[20],
    )


def _parse_trace(mm) -> TraceData:
    header = _parse_header(mm[:HEADER_SIZE])
    warp_buffer_size = WARP_BUFFER_HEADER_SIZE + header.events_per_warp * TRACE_EVENT_SIZE

    warps = []
    for w in range(header.total_warp_slots):
        offset = HEADER_SIZE + w * warp_buffer_size

        wh = struct.unpack_from(_WARP_HEADER_FMT, mm, offset)
        num_events = min(wh[2], header.events_per_warp)
        overflow_count = wh[1]

        events = []
        evt_offset = offset + WARP_BUFFER_HEADER_SIZE
        for e in range(num_events):
            ev = struct.unpack_from(_EVENT_FMT, mm, evt_offset + e * TRACE_EVENT_SIZE)
            events.append(TraceEvent(
                site_id=ev[0], event_type=ev[1], branch_dir=ev[2],
                active_mask=ev[4], value_a=ev[5],
            ))

        warps.append(WarpData(
            warp_idx=w, num_events=num_events,
            overflow_count=overflow_count, events=events,
        ))

    if header.has_history:
        history_offset = (
            header.history_section_offset
            if header.history_section_offset > 0
            else HEADER_SIZE + header.total_warp_slots * warp_buffer_size
        )
        ring_size = HISTORY_RING_HEADER_SIZE + header.history_depth * HISTORY_ENTRY_SIZE

        for w in range(header.total_warp_slots):
            ring_offset = history_offset + w * ring_size
            if ring_offset + ring_size > len(mm):
                break

            rh = struct.unpack_from(_HISTORY_RING_FMT, mm, ring_offset)
            depth, total_writes = rh[1], rh[2]
            if total_writes == 0:
                continue

            entries = []
            ent_offset = ring_offset + HISTORY_RING_HEADER_SIZE
            for e in range(min(depth, header.history_depth)):
                he = struct.unpack_from(_HISTORY_ENTRY_FMT, mm, ent_offset + e * HISTORY_ENTRY_SIZE)
                if total_writes <= depth and e >= total_writes:
                    break
                entries.append(HistoryEntry(site_id=he[0], value=he[1], seq=he[2]))

            entries.sort(key=lambda e: e.seq)
            warps[w].history = entries

    if header.has_snapshot:
        if header.snapshot_section_offset > 0:
            snap_offset = header.snapshot_section_offset
        else:
            base = HEADER_SIZE + header.total_warp_slots * warp_buffer_size
            if header.has_history:
                hist_ring_size = HISTORY_RING_HEADER_SIZE + header.history_depth * HISTORY_ENTRY_SIZE
                base += header.total_warp_slots * hist_ring_size
            snap_offset = base

        snap_ring_size = SNAPSHOT_RING_HEADER_SIZE + header.snapshot_depth * SNAPSHOT_ENTRY_SIZE

        for w in range(header.total_warp_slots):
            ring_offset = snap_offset + w * snap_ring_size
            if ring_offset + snap_ring_size > len(mm):
                break

            rh = struct.unpack_from(_SNAPSHOT_RING_FMT, mm, ring_offset)
            depth, total_writes = rh[1], rh[2]
            if total_writes == 0:
                continue

            entries = []
            ent_offset = ring_offset + SNAPSHOT_RING_HEADER_SIZE
            for e in range(min(depth, header.snapshot_depth)):
                if total_writes <= depth and e >= total_writes:
                    break
                se = struct.unpack_from(_SNAPSHOT_ENTRY_FMT, mm, ent_offset + e * SNAPSHOT_ENTRY_SIZE)
                entries.append(SnapshotEntry(
                    site_id=se[0], active_mask=se[1], seq=se[2], cmp_predicate=se[3],
                    lhs_values=list(se[4:36]), rhs_values=list(se[36:68]),
                ))

            entries.sort(key=lambda e: e.seq)
            warps[w].snapshots = entries

    return TraceData(header, warps)
