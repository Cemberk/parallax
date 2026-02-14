//! Integration tests for the differential analysis engine

use prlx_diff::differ::{diff_traces, DiffConfig, DivergenceKind};
use prlx_diff::trace_format::*;
use std::io::Write;
use bytemuck;
use tempfile::NamedTempFile;

/// Helper: Create a minimal valid trace file with given events
fn create_test_trace(kernel_name: &str, events: &[Vec<TraceEvent>]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    // Create header
    let mut header = TraceFileHeader {
        magic: PRLX_MAGIC,
        version: PRLX_VERSION,
        flags: 0,
        kernel_name_hash: 0x1234567890ABCDEF,
        kernel_name: [0; 64],
        grid_dim: [1, 1, 1],
        block_dim: [32, 1, 1],
        num_warps_per_block: 1,
        total_warp_slots: events.len() as u32,
        events_per_warp: PRLX_EVENTS_PER_WARP as u32,
        _pad: 0,
        timestamp: 1234567890,
        cuda_arch: 80,
        history_depth: 0,
        history_section_offset: 0,
        sample_rate: 0,
        snapshot_depth: 0,
        snapshot_section_offset: 0,
    };

    // Copy kernel name
    let name_bytes = kernel_name.as_bytes();
    let copy_len = name_bytes.len().min(63);
    header.kernel_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

    // Write header
    let header_bytes: &[u8] = bytemuck::bytes_of(&header);
    file.write_all(header_bytes).unwrap();

    // Write warp buffers
    for warp_events in events {
        // Write warp header
        let warp_header = WarpBufferHeader {
            write_idx: warp_events.len() as u32,
            overflow_count: 0,
            num_events: warp_events.len() as u32,
            total_event_count: 0,
        };
        let warp_header_bytes: &[u8] = bytemuck::bytes_of(&warp_header);
        file.write_all(warp_header_bytes).unwrap();

        // Write events
        for event in warp_events {
            let event_bytes: &[u8] = bytemuck::bytes_of(event);
            file.write_all(event_bytes).unwrap();
        }

        // Pad remaining events with zeros
        let remaining = PRLX_EVENTS_PER_WARP - warp_events.len();
        let zero_event = TraceEvent {
            site_id: 0,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0,
            value_a: 0,
        };
        for _ in 0..remaining {
            let zero_bytes: &[u8] = bytemuck::bytes_of(&zero_event);
            file.write_all(zero_bytes).unwrap();
        }
    }

    file.flush().unwrap();
    file
}

/// Test 1: Identical traces should produce no divergences
#[test]
fn test_identical_traces() {
    let events = vec![
        vec![
            TraceEvent {
                site_id: 0x1000,
                event_type: 0,
                branch_dir: 1,
                _reserved: 0,
                active_mask: 0xFFFFFFFF,
                value_a: 42,
            },
            TraceEvent {
                site_id: 0x2000,
                event_type: 0,
                branch_dir: 0,
                _reserved: 0,
                active_mask: 0xFFFFFFFF,
                value_a: 100,
            },
        ],
        vec![
            TraceEvent {
                site_id: 0x1000,
                event_type: 0,
                branch_dir: 0,
                _reserved: 0,
                active_mask: 0xFF00FF00,
                value_a: 99,
            },
        ],
    ];

    let trace_a_file = create_test_trace("test_kernel", &events);
    let trace_b_file = create_test_trace("test_kernel", &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert!(result.is_identical(), "Identical traces should have no divergences");
    assert_eq!(result.divergences.len(), 0);
    assert_eq!(result.warps_diverged, 0);
}

/// Test 2: Active mask mismatch should be detected
#[test]
fn test_active_mask_divergence() {
    let events_a = vec![vec![
        TraceEvent {
            site_id: 0x1000,
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF, // All threads active
            value_a: 42,
        },
        TraceEvent {
            site_id: 0x2000,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0x0000FFFF, // First 16 threads
            value_a: 100,
        },
    ]];

    let events_b = vec![vec![
        TraceEvent {
            site_id: 0x1000,
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 42,
        },
        TraceEvent {
            site_id: 0x2000,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFF0000, // Last 16 threads (DIFFERENT!)
            value_a: 100,
        },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert!(!result.is_identical());
    assert_eq!(result.divergences.len(), 1);
    assert_eq!(result.warps_diverged, 1);

    let div = &result.divergences[0];
    assert_eq!(div.warp_idx, 0);
    assert_eq!(div.event_idx, 1);
    assert_eq!(div.site_id, 0x2000);

    match div.kind {
        DivergenceKind::ActiveMask { mask_a, mask_b } => {
            assert_eq!(mask_a, 0x0000FFFF);
            assert_eq!(mask_b, 0xFFFF0000);
        }
        _ => panic!("Expected ActiveMask divergence, got {:?}", div.kind),
    }
}

/// Test 3: Extra events (drift) should be detected and re-synced
#[test]
fn test_extra_events_resync() {
    // Trace A: 3 events
    let events_a = vec![vec![
        TraceEvent {
            site_id: 0x1000,
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 1,
        },
        TraceEvent {
            site_id: 0x2000,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 2,
        },
        TraceEvent {
            site_id: 0x5000, // Final event
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 5,
        },
    ]];

    // Trace B: 5 events (2 extra in the middle)
    let events_b = vec![vec![
        TraceEvent {
            site_id: 0x1000,
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 1,
        },
        TraceEvent {
            site_id: 0x2000,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 2,
        },
        // Extra events (loop iterations)
        TraceEvent {
            site_id: 0x3000,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 3,
        },
        TraceEvent {
            site_id: 0x4000,
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 4,
        },
        TraceEvent {
            site_id: 0x5000, // Should re-sync here
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 5,
        },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert!(!result.is_identical());

    // Should detect ExtraEvents divergence
    let extra_event_divs: Vec<_> = result
        .divergences
        .iter()
        .filter(|d| matches!(d.kind, DivergenceKind::ExtraEvents { .. }))
        .collect();

    assert!(!extra_event_divs.is_empty(), "Should detect extra events");

    // Verify it's detected as extra events in trace B
    match &extra_event_divs[0].kind {
        DivergenceKind::ExtraEvents { count, in_trace_b } => {
            assert_eq!(*in_trace_b, true, "Extra events should be in trace B");
            assert_eq!(*count, 2, "Should detect 2 extra events");
        }
        _ => panic!("Expected ExtraEvents divergence"),
    }
}

/// Test 4: Branch direction divergence
#[test]
fn test_branch_direction_divergence() {
    let events_a = vec![vec![TraceEvent {
        site_id: 0x1000,
        event_type: 0,
        branch_dir: 0, // NOT-TAKEN
        _reserved: 0,
        active_mask: 0xFFFFFFFF,
        value_a: 42,
    }]];

    let events_b = vec![vec![TraceEvent {
        site_id: 0x1000,
        event_type: 0,
        branch_dir: 1, // TAKEN (DIFFERENT!)
        _reserved: 0,
        active_mask: 0xFFFFFFFF,
        value_a: 42,
    }]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert!(!result.is_identical());
    assert_eq!(result.divergences.len(), 1);

    match result.divergences[0].kind {
        DivergenceKind::Branch { dir_a, dir_b } => {
            assert_eq!(dir_a, 0);
            assert_eq!(dir_b, 1);
        }
        _ => panic!("Expected Branch divergence"),
    }
}

/// Test 5: True path divergence (different sites, no re-sync possible)
#[test]
fn test_path_divergence() {
    let events_a = vec![vec![
        TraceEvent {
            site_id: 0x1000,
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 1,
        },
        TraceEvent {
            site_id: 0xAAAA, // Path A
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 10,
        },
    ]];

    let events_b = vec![vec![
        TraceEvent {
            site_id: 0x1000,
            event_type: 0,
            branch_dir: 1,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 1,
        },
        TraceEvent {
            site_id: 0xBBBB, // Path B (completely different, no re-sync)
            event_type: 0,
            branch_dir: 0,
            _reserved: 0,
            active_mask: 0xFFFFFFFF,
            value_a: 20,
        },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig {
        lookahead_window: 2, // Small window, won't find re-sync
        ..Default::default()
    };
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert!(!result.is_identical());

    // Should detect true path divergence
    let path_divs: Vec<_> = result
        .divergences
        .iter()
        .filter(|d| matches!(d.kind, DivergenceKind::Path { .. }))
        .collect();

    assert!(!path_divs.is_empty(), "Should detect path divergence");

    match &path_divs[0].kind {
        DivergenceKind::Path { site_a, site_b } => {
            assert_eq!(*site_a, 0xAAAA);
            assert_eq!(*site_b, 0xBBBB);
        }
        _ => panic!("Expected Path divergence"),
    }
}

/// Test 6: Value comparison (when enabled)
#[test]
fn test_value_divergence() {
    let events_a = vec![vec![TraceEvent {
        site_id: 0x1000,
        event_type: 0,
        branch_dir: 1,
        _reserved: 0,
        active_mask: 0xFFFFFFFF,
        value_a: 100, // Value differs
    }]];

    let events_b = vec![vec![TraceEvent {
        site_id: 0x1000,
        event_type: 0,
        branch_dir: 1,
        _reserved: 0,
        active_mask: 0xFFFFFFFF,
        value_a: 200, // Different value
    }]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    // Test with value comparison disabled (default)
    let config_no_values = DiffConfig::default();
    let result_no_values = diff_traces(&trace_a, &trace_b, &config_no_values).unwrap();
    assert!(result_no_values.is_identical(), "Should be identical when not comparing values");

    // Test with value comparison enabled
    let config_with_values = DiffConfig {
        compare_values: true,
        ..Default::default()
    };
    let result_with_values = diff_traces(&trace_a, &trace_b, &config_with_values).unwrap();
    assert!(!result_with_values.is_identical(), "Should detect value divergence");

    match result_with_values.divergences[0].kind {
        DivergenceKind::Value { val_a, val_b } => {
            assert_eq!(val_a, 100);
            assert_eq!(val_b, 200);
        }
        _ => panic!("Expected Value divergence"),
    }
}

/// Helper: Create a trace file with snapshot data attached.
/// The snapshot section follows directly after the event buffers.
fn create_trace_with_snapshots(
    kernel_name: &str,
    events: &[Vec<TraceEvent>],
    snapshot_depth: u32,
    snapshots: &[Vec<SnapshotEntry>],
) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    let num_warps = events.len() as u32;

    let warp_buffer_size = std::mem::size_of::<WarpBufferHeader>()
        + PRLX_EVENTS_PER_WARP * std::mem::size_of::<TraceEvent>();
    let event_section_size = num_warps as usize * warp_buffer_size;
    let snap_ring_size =
        std::mem::size_of::<SnapshotRingHeader>() + snapshot_depth as usize * std::mem::size_of::<SnapshotEntry>();

    let mut header = TraceFileHeader {
        magic: PRLX_MAGIC,
        version: PRLX_VERSION,
        flags: PRLX_FLAG_SNAPSHOT,
        kernel_name_hash: 0xDEADBEEF,
        kernel_name: [0; 64],
        grid_dim: [1, 1, 1],
        block_dim: [32, 1, 1],
        num_warps_per_block: 1,
        total_warp_slots: num_warps,
        events_per_warp: PRLX_EVENTS_PER_WARP as u32,
        _pad: 0,
        timestamp: 999,
        cuda_arch: 80,
        history_depth: 0,
        history_section_offset: 0,
        sample_rate: 0,
        snapshot_depth,
        snapshot_section_offset: (std::mem::size_of::<TraceFileHeader>() + event_section_size) as u32,
    };

    let name_bytes = kernel_name.as_bytes();
    let copy_len = name_bytes.len().min(63);
    header.kernel_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

    file.write_all(bytemuck::bytes_of(&header)).unwrap();

    // Write warp event buffers
    for warp_events in events {
        let warp_header = WarpBufferHeader {
            write_idx: warp_events.len() as u32,
            overflow_count: 0,
            num_events: warp_events.len() as u32,
            total_event_count: 0,
        };
        file.write_all(bytemuck::bytes_of(&warp_header)).unwrap();

        for event in warp_events {
            file.write_all(bytemuck::bytes_of(event)).unwrap();
        }

        let remaining = PRLX_EVENTS_PER_WARP - warp_events.len();
        let zero_event = TraceEvent {
            site_id: 0, event_type: 0, branch_dir: 0, _reserved: 0,
            active_mask: 0, value_a: 0,
        };
        for _ in 0..remaining {
            file.write_all(bytemuck::bytes_of(&zero_event)).unwrap();
        }
    }

    // Write snapshot section (one ring per warp)
    for (w, warp_snaps) in snapshots.iter().enumerate() {
        let ring_header = SnapshotRingHeader {
            write_idx: warp_snaps.len() as u32,
            depth: snapshot_depth,
            total_writes: warp_snaps.len() as u32,
            _reserved: 0,
        };
        file.write_all(bytemuck::bytes_of(&ring_header)).unwrap();

        for snap in warp_snaps {
            file.write_all(bytemuck::bytes_of(snap)).unwrap();
        }

        // Pad remaining slots with zeros
        let remaining = snapshot_depth as usize - warp_snaps.len();
        let zero_snap = SnapshotEntry {
            site_id: 0, active_mask: 0, seq: 0, cmp_predicate: 0,
            lhs_values: [0; 32], rhs_values: [0; 32], _pad: [0; 4],
        };
        for _ in 0..remaining {
            file.write_all(bytemuck::bytes_of(&zero_snap)).unwrap();
        }
    }

    // Fill snapshot rings for warps that have events but no snapshot data
    for _ in snapshots.len()..events.len() {
        let ring_header = SnapshotRingHeader {
            write_idx: 0, depth: snapshot_depth, total_writes: 0, _reserved: 0,
        };
        file.write_all(bytemuck::bytes_of(&ring_header)).unwrap();
        let zero_snap = SnapshotEntry {
            site_id: 0, active_mask: 0, seq: 0, cmp_predicate: 0,
            lhs_values: [0; 32], rhs_values: [0; 32], _pad: [0; 4],
        };
        for _ in 0..snapshot_depth {
            file.write_all(bytemuck::bytes_of(&zero_snap)).unwrap();
        }
    }

    file.flush().unwrap();
    file
}

/// Test 7: Snapshot data flows through parser → differ → report
/// Verifies the full pipeline: trace with snapshot section → parser reads it →
/// differ attaches SnapshotContext to branch divergence → per-lane operands accessible.
#[test]
fn test_snapshot_integration() {
    let branch_site = 0x1000u32;

    // Both traces have same branch event at site 0x1000, but different directions
    let events_a = vec![vec![TraceEvent {
        site_id: branch_site,
        event_type: 0, // Branch
        branch_dir: 1, // TAKEN
        _reserved: 0,
        active_mask: 0xFFFFFFFF,
        value_a: 0,
    }]];

    let events_b = vec![vec![TraceEvent {
        site_id: branch_site,
        event_type: 0,
        branch_dir: 0, // NOT-TAKEN
        _reserved: 0,
        active_mask: 0xFFFFFFFF,
        value_a: 0,
    }]];

    // Snapshot A: threshold=10, all lanes compare their index against 10
    let mut snap_a = SnapshotEntry {
        site_id: branch_site,
        active_mask: 0xFFFFFFFF,
        seq: 0,
        cmp_predicate: 38, // ICMP_SGT
        lhs_values: [0; 32],
        rhs_values: [0; 32],
        _pad: [0; 4],
    };
    for i in 0..32 {
        snap_a.lhs_values[i] = i as u32;  // lane value = lane index
        snap_a.rhs_values[i] = 10;         // threshold = 10
    }

    // Snapshot B: threshold=64, same lane values
    let mut snap_b = SnapshotEntry {
        site_id: branch_site,
        active_mask: 0xFFFFFFFF,
        seq: 0,
        cmp_predicate: 38,
        lhs_values: [0; 32],
        rhs_values: [0; 32],
        _pad: [0; 4],
    };
    for i in 0..32 {
        snap_b.lhs_values[i] = i as u32;
        snap_b.rhs_values[i] = 64;         // threshold = 64
    }

    let file_a = create_trace_with_snapshots("test_kernel", &events_a, 4, &[vec![snap_a]]);
    let file_b = create_trace_with_snapshots("test_kernel", &events_b, 4, &[vec![snap_b]]);

    // Verify parser reads snapshot data
    let trace_a = prlx_diff::parser::TraceFile::open(file_a.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(file_b.path()).unwrap();

    assert!(trace_a.has_snapshot(), "Trace A should have snapshot data");
    assert!(trace_b.has_snapshot(), "Trace B should have snapshot data");

    // Verify parser can retrieve snapshot for warp 0
    let snap_a_parsed = trace_a.get_snapshot_for_site(0, branch_site).unwrap();
    assert!(snap_a_parsed.is_some(), "Should find snapshot for site in trace A");
    let snap_a_parsed = snap_a_parsed.unwrap();
    assert_eq!(snap_a_parsed.rhs_values[0], 10, "A: rhs lane 0 should be threshold=10");
    assert_eq!(snap_a_parsed.lhs_values[5], 5, "A: lhs lane 5 should be 5");

    let snap_b_parsed = trace_b.get_snapshot_for_site(0, branch_site).unwrap();
    assert!(snap_b_parsed.is_some());
    let snap_b_parsed = snap_b_parsed.unwrap();
    assert_eq!(snap_b_parsed.rhs_values[0], 64, "B: rhs lane 0 should be threshold=64");

    // Run the differ — it should detect branch divergence AND attach snapshot context
    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert!(!result.is_identical());
    assert_eq!(result.divergences.len(), 1);

    let div = &result.divergences[0];
    assert_eq!(div.site_id, branch_site);

    match div.kind {
        DivergenceKind::Branch { dir_a, dir_b } => {
            assert_eq!(dir_a, 1); // TAKEN
            assert_eq!(dir_b, 0); // NOT-TAKEN
        }
        _ => panic!("Expected Branch divergence, got {:?}", div.kind),
    }

    // The snapshot context should be attached
    let snap_ctx = div.snapshot.as_ref().expect("Divergence should have snapshot context");
    assert_eq!(snap_ctx.cmp_predicate, 38, "Should preserve ICmp predicate");

    // Verify per-lane operands came through
    assert_eq!(snap_ctx.rhs_a[0], 10, "Snapshot A rhs should be threshold=10");
    assert_eq!(snap_ctx.rhs_b[0], 64, "Snapshot B rhs should be threshold=64");
    assert_eq!(snap_ctx.lhs_a[15], 15, "Snapshot A lhs lane 15 should be 15");
    assert_eq!(snap_ctx.lhs_b[15], 15, "Snapshot B lhs lane 15 should be 15");

    // The key insight: A:rhs=10 vs B:rhs=64 explains the divergence
    for lane in 0..32 {
        assert_eq!(snap_ctx.lhs_a[lane], snap_ctx.lhs_b[lane],
            "LHS should be identical across traces (same input data)");
        assert_ne!(snap_ctx.rhs_a[lane], snap_ctx.rhs_b[lane],
            "RHS should differ (different threshold)");
    }
}
