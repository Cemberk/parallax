use prlx_diff::differ::{diff_traces, DiffConfig, DivergenceKind};
use prlx_diff::trace_format::*;
use std::io::Write;
use bytemuck;
use tempfile::NamedTempFile;

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

// ============================================================
// Error handling & validation tests
// ============================================================

fn create_test_trace_custom(
    kernel_name: &str,
    kernel_hash: u64,
    grid_dim: [u32; 3],
    block_dim: [u32; 3],
    events: &[Vec<TraceEvent>],
) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    let mut header = TraceFileHeader {
        magic: PRLX_MAGIC,
        version: PRLX_VERSION,
        flags: 0,
        kernel_name_hash: kernel_hash,
        kernel_name: [0; 64],
        grid_dim,
        block_dim,
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

    let name_bytes = kernel_name.as_bytes();
    let copy_len = name_bytes.len().min(63);
    header.kernel_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

    file.write_all(bytemuck::bytes_of(&header)).unwrap();

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

    file.flush().unwrap();
    file
}

#[test]
fn test_force_flag_allows_kernel_mismatch() {
    let events = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }]];

    let trace_a_file = create_test_trace_custom("kernel_a", 0xAAAA, [1,1,1], [32,1,1], &events);
    let trace_b_file = create_test_trace_custom("kernel_b", 0xBBBB, [1,1,1], [32,1,1], &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig { force: true, ..Default::default() };
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();
    assert!(result.is_identical(), "Identical events should produce no divergences with --force");
}

#[test]
fn test_kernel_name_mismatch_error() {
    let events = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }]];

    let trace_a_file = create_test_trace_custom("kernel_a", 0xAAAA, [1,1,1], [32,1,1], &events);
    let trace_b_file = create_test_trace_custom("kernel_b", 0xBBBB, [1,1,1], [32,1,1], &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default());
    assert!(result.is_err(), "Should error on kernel name mismatch without --force");
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Kernel mismatch"), "Error: {}", err_msg);
}

#[test]
fn test_grid_dim_mismatch_error() {
    let events = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }]];

    let trace_a_file = create_test_trace_custom("kern", 0x1234, [2,1,1], [32,1,1], &events);
    let trace_b_file = create_test_trace_custom("kern", 0x1234, [4,1,1], [32,1,1], &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default());
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Grid dimension mismatch"), "Error: {}", err_msg);
}

#[test]
fn test_block_dim_mismatch_error() {
    let events = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }]];

    let trace_a_file = create_test_trace_custom("kern", 0x1234, [1,1,1], [32,1,1], &events);
    let trace_b_file = create_test_trace_custom("kern", 0x1234, [1,1,1], [64,1,1], &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default());
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("Block dimension mismatch"), "Error: {}", err_msg);
}

// ============================================================
// Config option tests
// ============================================================

#[test]
fn test_max_divergences_limit() {
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
        TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
        TraceEvent { site_id: 0x4000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 4 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
        TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
        TraceEvent { site_id: 0x4000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 4 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig { max_divergences: 2, ..Default::default() };
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();

    assert_eq!(result.divergences.len(), 2, "Should truncate to max_divergences=2");
}

#[test]
fn test_lookahead_window_size() {
    // Both traces share 0x1000 first, then B has 5 extra events before rejoining at 0x9000
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x9000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 9 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
        TraceEvent { site_id: 0x4000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 4 },
        TraceEvent { site_id: 0x5000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 5 },
        TraceEvent { site_id: 0x6000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 6 },
        TraceEvent { site_id: 0x7000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 7 },
        TraceEvent { site_id: 0x9000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 9 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    // After matching 0x1000, A has 0x9000, B has 0x3000. Resync target (0x9000) is 5 events ahead in B.
    // With window=2, can't reach it → path divergence
    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();
    let result_small = diff_traces(&trace_a, &trace_b, &DiffConfig { lookahead_window: 2, ..Default::default() }).unwrap();
    assert!(result_small.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::Path { .. })),
        "Small lookahead should result in path divergence");

    // With window=32, finds 0x9000 → extra events detection
    let trace_a2 = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b2 = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();
    let result_large = diff_traces(&trace_a2, &trace_b2, &DiffConfig { lookahead_window: 32, ..Default::default() }).unwrap();
    assert!(result_large.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::ExtraEvents { .. })),
        "Large lookahead should detect extra events and resync");
    assert!(!result_large.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::Path { .. })),
        "Large lookahead should not produce path divergence");
}

// ============================================================
// Edge case tests
// ============================================================

#[test]
fn test_empty_warps() {
    let events: Vec<Vec<TraceEvent>> = vec![vec![], vec![]];

    let trace_a_file = create_test_trace("test_kernel", &events);
    let trace_b_file = create_test_trace("test_kernel", &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert!(result.is_identical(), "Empty warps should be identical");
    assert_eq!(result.total_events_a, 0);
    assert_eq!(result.total_events_b, 0);
    assert_eq!(result.total_warps, 2);
}

#[test]
fn test_single_event_traces() {
    let events = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }]];

    let trace_a_file = create_test_trace("test_kernel", &events);
    let trace_b_file = create_test_trace("test_kernel", &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert!(result.is_identical());
    assert_eq!(result.total_events_a, 1);
}

#[test]
fn test_multiple_warps_mixed() {
    let events_a = vec![
        vec![TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 }],
        vec![TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 }],
        vec![TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 }],
        vec![TraceEvent { site_id: 0x4000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 4 }],
    ];

    let events_b = vec![
        vec![TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 }],
        vec![TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 }], // dir flipped
        vec![TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 }],
        vec![TraceEvent { site_id: 0x4000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0x0000FFFF, value_a: 4 }], // mask differs
    ];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert_eq!(result.warps_diverged, 2, "Warps 1 and 3 should diverge");
    assert_eq!(result.divergences.len(), 2);
    assert_eq!(result.total_warps, 4);
}

#[test]
fn test_multiple_divergence_types_same_warp() {
    let events_a = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 100,
    }]];

    let events_b = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0x0000FFFF, value_a: 200,
    }]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert_eq!(result.divergences.len(), 2, "Branch + mask divergence");
    assert_eq!(result.warps_diverged, 1, "Both in same warp");
    assert!(result.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::Branch { .. })));
    assert!(result.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::ActiveMask { .. })));
}

#[test]
fn test_trailing_events_trace_a() {
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
        TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert!(!result.is_identical());

    let extra = result.divergences.iter().find(|d| matches!(d.kind, DivergenceKind::ExtraEvents { .. })).unwrap();
    match &extra.kind {
        DivergenceKind::ExtraEvents { count, in_trace_b } => {
            assert_eq!(*count, 2);
            assert!(!in_trace_b, "Extra events in trace A");
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_trailing_events_trace_b() {
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
        TraceEvent { site_id: 0x3000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert!(!result.is_identical());

    let extra = result.divergences.iter().find(|d| matches!(d.kind, DivergenceKind::ExtraEvents { .. })).unwrap();
    match &extra.kind {
        DivergenceKind::ExtraEvents { count, in_trace_b } => {
            assert_eq!(*count, 2);
            assert!(in_trace_b, "Extra events in trace B");
        }
        _ => unreachable!(),
    }
}

// ============================================================
// DiffResult API tests
// ============================================================

#[test]
fn test_divergences_by_site() {
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 3 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    let by_site = result.divergences_by_site();
    assert_eq!(by_site.len(), 2);
    assert_eq!(by_site[&0x1000].len(), 2);
    assert_eq!(by_site[&0x2000].len(), 1);
}

#[test]
fn test_divergences_by_kind() {
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0x0000FFFF, value_a: 2 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    let by_kind = result.divergences_by_kind();
    assert_eq!(by_kind["Branch"], 1);
    assert_eq!(by_kind["ActiveMask"], 1);
    assert_eq!(by_kind.len(), 2);
}

// ============================================================
// Cross-language format verification tests
// ============================================================

#[test]
fn test_header_size_matches_160_bytes() {
    assert_eq!(std::mem::size_of::<TraceFileHeader>(), 160);
}

#[test]
fn test_header_field_offsets() {
    let header = TraceFileHeader {
        magic: PRLX_MAGIC,
        version: PRLX_VERSION,
        flags: 0,
        kernel_name_hash: 0x1234567890ABCDEF,
        kernel_name: [0; 64],
        grid_dim: [1, 1, 1],
        block_dim: [32, 1, 1],
        num_warps_per_block: 1,
        total_warp_slots: 1,
        events_per_warp: 128,
        _pad: 0,
        timestamp: 0xDEADCAFEBABE0000,
        cuda_arch: 80,
        history_depth: 0,
        history_section_offset: 0,
        sample_rate: 0,
        snapshot_depth: 0,
        snapshot_section_offset: 0,
    };
    let bytes: &[u8] = bytemuck::bytes_of(&header);

    assert_eq!(&bytes[0..8], &PRLX_MAGIC.to_le_bytes());
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), PRLX_VERSION);
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0x1234567890ABCDEF);
    assert_eq!(&bytes[24..88], &[0u8; 64]);
    assert_eq!(u32::from_le_bytes(bytes[120..124].try_into().unwrap()), 128);
    assert_eq!(&bytes[124..128], &[0u8; 4]); // padding before timestamp
    assert_eq!(u64::from_le_bytes(bytes[128..136].try_into().unwrap()), 0xDEADCAFEBABE0000);
    assert_eq!(u32::from_le_bytes(bytes[136..140].try_into().unwrap()), 80);
}

#[test]
fn test_event_byte_layout() {
    let event = TraceEvent {
        site_id: 0xABCD1234,
        event_type: 2,
        branch_dir: 1,
        _reserved: 0,
        active_mask: 0xFFFF0000,
        value_a: 0xDEADBEEF,
    };
    let bytes: &[u8] = bytemuck::bytes_of(&event);
    assert_eq!(bytes.len(), 16);

    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0xABCD1234);
    assert_eq!(bytes[4], 2);
    assert_eq!(bytes[5], 1);
    assert_eq!(&bytes[6..8], &[0, 0]);
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0xFFFF0000);
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 0xDEADBEEF);
}

#[test]
fn test_small_events_per_warp() {
    let events_per_warp = 128usize;
    let events = vec![vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }]];

    let mut file = NamedTempFile::new().unwrap();
    let mut header = TraceFileHeader {
        magic: PRLX_MAGIC,
        version: PRLX_VERSION,
        flags: 0,
        kernel_name_hash: 0x1234567890ABCDEF,
        kernel_name: [0; 64],
        grid_dim: [1, 1, 1],
        block_dim: [32, 1, 1],
        num_warps_per_block: 1,
        total_warp_slots: 1,
        events_per_warp: events_per_warp as u32,
        _pad: 0,
        timestamp: 1234567890,
        cuda_arch: 80,
        history_depth: 0,
        history_section_offset: 0,
        sample_rate: 0,
        snapshot_depth: 0,
        snapshot_section_offset: 0,
    };
    let name = b"test_kernel";
    header.kernel_name[..name.len()].copy_from_slice(name);
    file.write_all(bytemuck::bytes_of(&header)).unwrap();

    let warp_header = WarpBufferHeader {
        write_idx: 1, overflow_count: 0, num_events: 1, total_event_count: 0,
    };
    file.write_all(bytemuck::bytes_of(&warp_header)).unwrap();
    file.write_all(bytemuck::bytes_of(&events[0][0])).unwrap();
    let zero_event = TraceEvent {
        site_id: 0, event_type: 0, branch_dir: 0, _reserved: 0,
        active_mask: 0, value_a: 0,
    };
    for _ in 0..events_per_warp - 1 {
        file.write_all(bytemuck::bytes_of(&zero_event)).unwrap();
    }
    file.flush().unwrap();

    let trace = prlx_diff::parser::TraceFile::open(file.path()).unwrap();
    assert_eq!(trace.header().events_per_warp, 128);
    assert_eq!(trace.total_events(), 1);
    let (wh, evts) = trace.get_warp_data(0).unwrap();
    assert_eq!(wh.num_events, 1);
    assert_eq!(evts[0].site_id, 0x1000);
    assert_eq!(evts[0].value_a, 42);
}

// ============================================================
// Ring buffer overflow tests
// ============================================================

fn create_test_trace_with_overflow(
    events: &[TraceEvent],
    overflow_count: u32,
) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    let mut header = TraceFileHeader {
        magic: PRLX_MAGIC,
        version: PRLX_VERSION,
        flags: 0,
        kernel_name_hash: 0x1234567890ABCDEF,
        kernel_name: [0; 64],
        grid_dim: [1, 1, 1],
        block_dim: [32, 1, 1],
        num_warps_per_block: 1,
        total_warp_slots: 1,
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
    let name = b"test_kernel";
    header.kernel_name[..name.len()].copy_from_slice(name);
    file.write_all(bytemuck::bytes_of(&header)).unwrap();

    let actual_written = events.len().min(PRLX_EVENTS_PER_WARP);
    let warp_header = WarpBufferHeader {
        write_idx: (events.len() as u32) + overflow_count,
        overflow_count,
        num_events: actual_written as u32,
        total_event_count: 0,
    };
    file.write_all(bytemuck::bytes_of(&warp_header)).unwrap();

    for event in events.iter().take(PRLX_EVENTS_PER_WARP) {
        file.write_all(bytemuck::bytes_of(event)).unwrap();
    }
    let zero_event = TraceEvent {
        site_id: 0, event_type: 0, branch_dir: 0, _reserved: 0,
        active_mask: 0, value_a: 0,
    };
    for _ in 0..PRLX_EVENTS_PER_WARP - actual_written {
        file.write_all(bytemuck::bytes_of(&zero_event)).unwrap();
    }

    file.flush().unwrap();
    file
}

#[test]
fn test_overflow_count_reported() {
    let events = vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }];

    let file = create_test_trace_with_overflow(&events, 500);
    let trace = prlx_diff::parser::TraceFile::open(file.path()).unwrap();

    assert_eq!(trace.total_overflows(), 500);
    assert_eq!(trace.total_events(), 1);
    let (wh, evts) = trace.get_warp_data(0).unwrap();
    assert_eq!(wh.overflow_count, 500);
    assert_eq!(evts.len(), 1);
}

#[test]
fn test_overflow_with_full_buffer() {
    let mut events = Vec::new();
    for i in 0..PRLX_EVENTS_PER_WARP {
        events.push(TraceEvent {
            site_id: i as u32, event_type: 0, branch_dir: 1, _reserved: 0,
            active_mask: 0xFFFFFFFF, value_a: i as u32,
        });
    }

    let file = create_test_trace_with_overflow(&events, 1000);
    let trace = prlx_diff::parser::TraceFile::open(file.path()).unwrap();

    assert_eq!(trace.total_events(), PRLX_EVENTS_PER_WARP);
    assert_eq!(trace.total_overflows(), 1000);
    let (wh, evts) = trace.get_warp_data(0).unwrap();
    assert_eq!(wh.num_events as usize, PRLX_EVENTS_PER_WARP);
    assert_eq!(evts.len(), PRLX_EVENTS_PER_WARP);
    assert_eq!(evts[0].site_id, 0);
    assert_eq!(evts[PRLX_EVENTS_PER_WARP - 1].site_id, (PRLX_EVENTS_PER_WARP - 1) as u32);
}

#[test]
fn test_overflow_diff_identical() {
    let events = vec![TraceEvent {
        site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0,
        active_mask: 0xFFFFFFFF, value_a: 42,
    }];

    let file_a = create_test_trace_with_overflow(&events, 100);
    let file_b = create_test_trace_with_overflow(&events, 200);

    let trace_a = prlx_diff::parser::TraceFile::open(file_a.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(file_b.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert!(result.is_identical());
}

// ============================================================
// Zstd compression tests
// ============================================================

#[test]
fn test_compressed_trace_roundtrip() {
    let events = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 42 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0x0000FFFF, value_a: 99 },
    ]];

    let uncompressed_file = create_test_trace("test_kernel", &events);
    let uncompressed_data = std::fs::read(uncompressed_file.path()).unwrap();
    let header_size = std::mem::size_of::<TraceFileHeader>();

    let mut header: TraceFileHeader = *bytemuck::from_bytes(&uncompressed_data[..header_size]);
    header.flags |= PRLX_FLAG_COMPRESS;

    let payload = &uncompressed_data[header_size..];
    let compressed = zstd::encode_all(payload, 3).unwrap();

    let mut compressed_file = NamedTempFile::new().unwrap();
    compressed_file.write_all(bytemuck::bytes_of(&header)).unwrap();
    compressed_file.write_all(&compressed).unwrap();
    compressed_file.flush().unwrap();

    let trace = prlx_diff::parser::TraceFile::open(compressed_file.path()).unwrap();
    assert_eq!(trace.total_events(), 2);
    let (_, evts) = trace.get_warp_data(0).unwrap();
    assert_eq!(evts[0].site_id, 0x1000);
    assert_eq!(evts[0].value_a, 42);
    assert_eq!(evts[1].site_id, 0x2000);
    assert_eq!(evts[1].active_mask, 0x0000FFFF);
}

#[test]
fn test_compressed_trace_diff() {
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ]];
    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ]];

    let header_size = std::mem::size_of::<TraceFileHeader>();

    let mut make_compressed = |events: &[Vec<TraceEvent>]| -> NamedTempFile {
        let raw_file = create_test_trace("test_kernel", events);
        let raw_data = std::fs::read(raw_file.path()).unwrap();
        let mut header: TraceFileHeader = *bytemuck::from_bytes(&raw_data[..header_size]);
        header.flags |= PRLX_FLAG_COMPRESS;
        let compressed = zstd::encode_all(&raw_data[header_size..], 3).unwrap();
        let mut out = NamedTempFile::new().unwrap();
        out.write_all(bytemuck::bytes_of(&header)).unwrap();
        out.write_all(&compressed).unwrap();
        out.flush().unwrap();
        out
    };

    let file_a = make_compressed(&events_a);
    let file_b = make_compressed(&events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(file_a.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(file_b.path()).unwrap();

    let result = diff_traces(&trace_a, &trace_b, &DiffConfig::default()).unwrap();
    assert!(!result.is_identical());
    assert!(result.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::Branch { .. })));
}

// ============================================================
// Session diff tests
// ============================================================

use prlx_diff::differ::diff_session;
use prlx_diff::parser::{SessionManifest, SessionLaunch};

/// Helper: create a session directory with per-kernel trace files + session.json manifest.
/// `launches` is a slice of (kernel_name, launch_index, events_per_warp_vec).
fn create_test_session(
    launches: &[(&str, u32, &[Vec<TraceEvent>])],
) -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().unwrap();

    let mut manifest_entries = Vec::new();

    for (kernel, launch_idx, events) in launches {
        let filename = format!("{}_{}.prlx", kernel, launch_idx);
        let trace_path = dir.path().join(&filename);

        // Write trace file
        let tmp = create_test_trace(kernel, events);
        std::fs::copy(tmp.path(), &trace_path).unwrap();

        // Determine grid/block from the trace
        manifest_entries.push(serde_json::json!({
            "kernel": kernel,
            "launch": launch_idx,
            "file": filename,
            "grid": [1, 1, 1],
            "block": [32, 1, 1],
        }));
    }

    let manifest_path = dir.path().join("session.json");
    let json = serde_json::to_string_pretty(&manifest_entries).unwrap();
    std::fs::write(&manifest_path, json).unwrap();

    dir
}

#[test]
fn test_session_identical() {
    let events = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ];

    let session_a = create_test_session(&[
        ("scale_kernel", 0, &[events.clone()]),
        ("threshold_kernel", 1, &[events.clone()]),
    ]);
    let session_b = create_test_session(&[
        ("scale_kernel", 0, &[events.clone()]),
        ("threshold_kernel", 1, &[events.clone()]),
    ]);

    let manifest_a = SessionManifest::load(session_a.path()).unwrap();
    let manifest_b = SessionManifest::load(session_b.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_session(&manifest_a, &manifest_b, &config);

    assert_eq!(result.kernel_results.len(), 2);
    assert!(result.unmatched_a.is_empty());
    assert!(result.unmatched_b.is_empty());
    for (_, _, diff_result) in &result.kernel_results {
        assert!(diff_result.as_ref().unwrap().is_identical());
    }
}

#[test]
fn test_session_branch_divergence() {
    let events_a = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ];
    let events_b = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ];

    let session_a = create_test_session(&[
        ("my_kernel", 0, &[events_a]),
    ]);
    let session_b = create_test_session(&[
        ("my_kernel", 0, &[events_b]),
    ]);

    let manifest_a = SessionManifest::load(session_a.path()).unwrap();
    let manifest_b = SessionManifest::load(session_b.path()).unwrap();

    let result = diff_session(&manifest_a, &manifest_b, &DiffConfig::default());

    assert_eq!(result.kernel_results.len(), 1);
    let (name, idx, diff_result) = &result.kernel_results[0];
    assert_eq!(name, "my_kernel");
    assert_eq!(*idx, 0);
    let diff = diff_result.as_ref().unwrap();
    assert!(!diff.is_identical());
    assert!(diff.divergences.iter().any(|d| matches!(d.kind, DivergenceKind::Branch { .. })));
}

#[test]
fn test_session_unmatched_launch() {
    let events = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ];

    // Session A has kernel_c that B doesn't
    let session_a = create_test_session(&[
        ("kernel_a", 0, &[events.clone()]),
        ("kernel_c", 1, &[events.clone()]),
    ]);
    let session_b = create_test_session(&[
        ("kernel_a", 0, &[events.clone()]),
    ]);

    let manifest_a = SessionManifest::load(session_a.path()).unwrap();
    let manifest_b = SessionManifest::load(session_b.path()).unwrap();

    let result = diff_session(&manifest_a, &manifest_b, &DiffConfig::default());

    assert_eq!(result.kernel_results.len(), 1); // Only kernel_a matched
    assert_eq!(result.unmatched_a.len(), 1);
    assert!(result.unmatched_a[0].contains("kernel_c"));
    assert!(result.unmatched_b.is_empty());
}

#[test]
fn test_session_different_launch_counts() {
    let events = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ];

    // A has 3 launches, B has 2 — launch 2 is unmatched in A
    let session_a = create_test_session(&[
        ("kern", 0, &[events.clone()]),
        ("kern", 1, &[events.clone()]),
        ("kern", 2, &[events.clone()]),
    ]);
    let session_b = create_test_session(&[
        ("kern", 0, &[events.clone()]),
        ("kern", 1, &[events.clone()]),
    ]);

    let manifest_a = SessionManifest::load(session_a.path()).unwrap();
    let manifest_b = SessionManifest::load(session_b.path()).unwrap();

    let result = diff_session(&manifest_a, &manifest_b, &DiffConfig::default());

    assert_eq!(result.kernel_results.len(), 2);
    assert_eq!(result.unmatched_a.len(), 1);
    assert!(result.unmatched_a[0].contains("kern"));
    assert!(result.unmatched_a[0].contains("launch 2"));
    assert!(result.unmatched_b.is_empty());
}

#[test]
fn test_session_manifest_parse() {
    let events = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 42 },
    ];

    let session = create_test_session(&[
        ("scale_kernel", 0, &[events.clone()]),
        ("reduce_kernel", 1, &[events.clone()]),
    ]);

    let manifest = SessionManifest::load(session.path()).unwrap();

    assert_eq!(manifest.launches.len(), 2);
    assert_eq!(manifest.launches[0].kernel, "scale_kernel");
    assert_eq!(manifest.launches[0].launch, 0);
    assert_eq!(manifest.launches[1].kernel, "reduce_kernel");
    assert_eq!(manifest.launches[1].launch, 1);
    assert_eq!(manifest.launches[0].grid, [1, 1, 1]);
    assert_eq!(manifest.launches[0].block, [32, 1, 1]);
}

#[test]
fn test_session_missing_trace_file() {
    let dir = tempfile::TempDir::new().unwrap();

    // Write manifest referencing a file that doesn't exist
    let manifest_json = serde_json::json!([{
        "kernel": "missing_kernel",
        "launch": 0,
        "file": "nonexistent.prlx",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
    }]);
    let manifest_path = dir.path().join("session.json");
    std::fs::write(&manifest_path, serde_json::to_string(&manifest_json).unwrap()).unwrap();

    // Create a second valid session to diff against
    let events = vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
    ];
    let session_b = create_test_session(&[("missing_kernel", 0, &[events])]);

    let manifest_a = SessionManifest::load(dir.path()).unwrap();
    let manifest_b = SessionManifest::load(session_b.path()).unwrap();

    let result = diff_session(&manifest_a, &manifest_b, &DiffConfig::default());

    // The matching launch should produce an error (file not found)
    assert_eq!(result.kernel_results.len(), 1);
    let (_, _, diff_result) = &result.kernel_results[0];
    assert!(diff_result.is_err(), "Should error when trace file is missing");
}

// ============================================================
// JSON output tests
// ============================================================

use prlx_diff::json_output::format_json_report;
use prlx_diff::differ::{Divergence, DiffResult};

/// Helper: build a DiffResult manually from a list of divergences.
fn make_diff_result(divergences: Vec<Divergence>, total_warps: usize) -> DiffResult {
    let warps_diverged = divergences
        .iter()
        .map(|d| d.warp_idx)
        .collect::<std::collections::HashSet<_>>()
        .len();
    DiffResult {
        divergences,
        total_warps,
        total_events_a: 10,
        total_events_b: 10,
        warps_compared: total_warps,
        warps_diverged,
        cross_gpu_info: None,
    }
}

#[test]
fn test_json_output_identical() {
    let result = make_diff_result(vec![], 4);
    let report = format_json_report(&result, None, None, false);

    assert!(report.passed, "No divergences should mean passed=true");
    assert_eq!(report.status, "identical");
    assert_eq!(report.total_divergences, 0);
    assert_eq!(report.counted_divergences, 0);
    assert_eq!(report.warps_compared, 4);
    assert_eq!(report.warps_diverged, 0);
    assert!(report.divergences.is_empty());
    assert_eq!(report.threshold, Some(0));

    // Verify it serializes to valid JSON
    let json_str = serde_json::to_string(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["status"], "identical");
    assert_eq!(parsed["passed"], true);
}

#[test]
fn test_json_output_diverged() {
    let divergences = vec![
        Divergence {
            warp_idx: 0,
            event_idx: 1,
            site_id: 0x1000,
            kind: DivergenceKind::Branch { dir_a: 0, dir_b: 1 },
            snapshot: None,
        },
        Divergence {
            warp_idx: 1,
            event_idx: 0,
            site_id: 0x2000,
            kind: DivergenceKind::ActiveMask { mask_a: 0xFFFFFFFF, mask_b: 0x0000FFFF },
            snapshot: None,
        },
    ];
    let result = make_diff_result(divergences, 4);
    let report = format_json_report(&result, None, None, false);

    assert!(!report.passed, "Divergences with no threshold should mean passed=false");
    assert_eq!(report.status, "diverged");
    assert_eq!(report.total_divergences, 2);
    assert_eq!(report.counted_divergences, 2);
    assert_eq!(report.warps_diverged, 2);
    assert_eq!(report.divergences.len(), 2);
    assert_eq!(report.threshold, Some(0));

    // Check individual divergence entries
    assert_eq!(report.divergences[0].warp_idx, 0);
    assert_eq!(report.divergences[0].site_id, "0x00001000");
    assert_eq!(report.divergences[0].kind, "Branch");
    assert_eq!(report.divergences[1].kind, "ActiveMask");

    // Verify breakdown
    assert_eq!(report.divergence_breakdown["Branch"], 1);
    assert_eq!(report.divergence_breakdown["ActiveMask"], 1);

    // Verify serialization
    let json_str = serde_json::to_string(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["passed"], false);
    assert_eq!(parsed["status"], "diverged");
}

#[test]
fn test_json_max_allowed_divergences() {
    let divergences = vec![
        Divergence {
            warp_idx: 0,
            event_idx: 0,
            site_id: 0x1000,
            kind: DivergenceKind::Branch { dir_a: 0, dir_b: 1 },
            snapshot: None,
        },
        Divergence {
            warp_idx: 0,
            event_idx: 1,
            site_id: 0x2000,
            kind: DivergenceKind::Branch { dir_a: 1, dir_b: 0 },
            snapshot: None,
        },
        Divergence {
            warp_idx: 1,
            event_idx: 0,
            site_id: 0x3000,
            kind: DivergenceKind::ActiveMask { mask_a: 0xFFFFFFFF, mask_b: 0x00FF00FF },
            snapshot: None,
        },
    ];
    let result = make_diff_result(divergences, 4);

    // threshold=5, counted=3 -> passed=true
    let report = format_json_report(&result, None, Some(5), false);
    assert!(report.passed, "3 divergences <= threshold 5 should pass");
    assert_eq!(report.counted_divergences, 3);
    assert_eq!(report.total_divergences, 3);
    assert_eq!(report.threshold, Some(5));
    assert_eq!(report.status, "diverged"); // status reflects actual state, not threshold

    // threshold=2, counted=3 -> passed=false
    let report_fail = format_json_report(&result, None, Some(2), false);
    assert!(!report_fail.passed, "3 divergences > threshold 2 should fail");

    // threshold=3, counted=3 -> passed=true (<=, not <)
    let report_exact = format_json_report(&result, None, Some(3), false);
    assert!(report_exact.passed, "3 divergences <= threshold 3 should pass (boundary)");
}

#[test]
fn test_json_ignore_active_mask() {
    // Create result with ONLY ActiveMask divergences
    let divergences = vec![
        Divergence {
            warp_idx: 0,
            event_idx: 0,
            site_id: 0x1000,
            kind: DivergenceKind::ActiveMask { mask_a: 0xFFFFFFFF, mask_b: 0x0000FFFF },
            snapshot: None,
        },
        Divergence {
            warp_idx: 1,
            event_idx: 0,
            site_id: 0x2000,
            kind: DivergenceKind::ActiveMask { mask_a: 0xFF00FF00, mask_b: 0x00FF00FF },
            snapshot: None,
        },
    ];
    let result = make_diff_result(divergences, 4);

    // With ignore_active_mask=true and no threshold -> counted=0, passed=true
    let report = format_json_report(&result, None, None, true);
    assert!(report.passed, "Only ActiveMask divergences with ignore_active_mask should pass");
    assert_eq!(report.total_divergences, 2, "total_divergences should still count all");
    assert_eq!(report.counted_divergences, 0, "counted_divergences should exclude ActiveMask");
    assert_eq!(report.status, "diverged", "status reflects raw state before filtering");

    // All divergences should still appear in the output list
    assert_eq!(report.divergences.len(), 2);

    // Now test mixed divergences: 1 Branch + 2 ActiveMask, ignore_active_mask=true
    let mixed_divergences = vec![
        Divergence {
            warp_idx: 0,
            event_idx: 0,
            site_id: 0x1000,
            kind: DivergenceKind::Branch { dir_a: 0, dir_b: 1 },
            snapshot: None,
        },
        Divergence {
            warp_idx: 0,
            event_idx: 1,
            site_id: 0x2000,
            kind: DivergenceKind::ActiveMask { mask_a: 0xFFFFFFFF, mask_b: 0x0000FFFF },
            snapshot: None,
        },
        Divergence {
            warp_idx: 1,
            event_idx: 0,
            site_id: 0x3000,
            kind: DivergenceKind::ActiveMask { mask_a: 0xFF00FF00, mask_b: 0x00FF00FF },
            snapshot: None,
        },
    ];
    let mixed_result = make_diff_result(mixed_divergences, 4);

    // ignore_active_mask=true, no threshold -> counted=1 (only the Branch), passed=false
    let mixed_report = format_json_report(&mixed_result, None, None, true);
    assert!(!mixed_report.passed, "Branch divergence should still cause failure");
    assert_eq!(mixed_report.counted_divergences, 1);
    assert_eq!(mixed_report.total_divergences, 3);

    // ignore_active_mask=true, threshold=1 -> counted=1 <= 1, passed=true
    let mixed_report_threshold = format_json_report(&mixed_result, None, Some(1), true);
    assert!(mixed_report_threshold.passed, "1 counted divergence <= threshold 1 should pass");
}

// ============================================================
// Flamegraph export tests
// ============================================================

use prlx_diff::flamegraph::export_flamegraph;

#[test]
fn test_flamegraph_export_basic() {
    // Create two traces with branch divergence
    let events_a = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 2 },
    ]];

    let events_b = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 1 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0x0000FFFF, value_a: 2 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events_a);
    let trace_b_file = create_test_trace("test_kernel", &events_b);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();
    assert!(!result.is_identical(), "Precondition: traces should diverge");

    // Export flamegraph to a temp file
    let output_file = NamedTempFile::new().unwrap();
    let output_path = output_file.path().to_path_buf();
    export_flamegraph(&trace_a, &trace_b, &result, None, &output_path).unwrap();

    // Read the output and verify it is valid JSON with "traceEvents" key
    let content = std::fs::read_to_string(&output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(parsed.is_object(), "Output should be a JSON object");
    assert!(parsed.get("traceEvents").is_some(), "Should have traceEvents key");

    let trace_events = parsed["traceEvents"].as_array().unwrap();
    assert!(!trace_events.is_empty(), "Should have at least one trace event");

    // Verify metadata event exists (process_name)
    let metadata_events: Vec<_> = trace_events
        .iter()
        .filter(|e| e["ph"] == "M")
        .collect();
    assert!(!metadata_events.is_empty(), "Should have metadata events");

    // Verify duration events exist for divergences (ph == "X")
    let duration_events: Vec<_> = trace_events
        .iter()
        .filter(|e| e["ph"] == "X")
        .collect();
    assert!(!duration_events.is_empty(), "Should have duration events for divergences");

    // Each duration event should have required fields
    for evt in &duration_events {
        assert!(evt.get("name").is_some(), "Duration event should have name");
        assert!(evt.get("cat").is_some(), "Duration event should have category");
        assert!(evt.get("ts").is_some(), "Duration event should have timestamp");
        assert!(evt.get("dur").is_some(), "Duration event should have duration");
        assert!(evt.get("pid").is_some(), "Duration event should have pid (block)");
        assert!(evt.get("tid").is_some(), "Duration event should have tid (warp)");
        assert!(evt.get("args").is_some(), "Duration event should have args");
    }

    // Verify counter events exist (ph == "C")
    let counter_events: Vec<_> = trace_events
        .iter()
        .filter(|e| e["ph"] == "C")
        .collect();
    assert!(!counter_events.is_empty(), "Should have counter events for heatmap");

    // Verify event names contain divergence kind info
    let all_names: Vec<String> = duration_events
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect();
    assert!(
        all_names.iter().any(|n| n.contains("Branch")),
        "Should have a Branch divergence event, got: {:?}", all_names
    );
}

#[test]
fn test_flamegraph_empty_diff() {
    // Create identical traces -> no divergences
    let events = vec![vec![
        TraceEvent { site_id: 0x1000, event_type: 0, branch_dir: 1, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 42 },
        TraceEvent { site_id: 0x2000, event_type: 0, branch_dir: 0, _reserved: 0, active_mask: 0xFFFFFFFF, value_a: 99 },
    ]];

    let trace_a_file = create_test_trace("test_kernel", &events);
    let trace_b_file = create_test_trace("test_kernel", &events);

    let trace_a = prlx_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = prlx_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

    let config = DiffConfig::default();
    let result = diff_traces(&trace_a, &trace_b, &config).unwrap();
    assert!(result.is_identical(), "Precondition: traces should be identical");

    // Export flamegraph
    let output_file = NamedTempFile::new().unwrap();
    let output_path = output_file.path().to_path_buf();
    export_flamegraph(&trace_a, &trace_b, &result, None, &output_path).unwrap();

    // Read the output and verify structure
    let content = std::fs::read_to_string(&output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(parsed.is_object(), "Output should be a JSON object");
    assert!(parsed.get("traceEvents").is_some(), "Should have traceEvents key");

    let trace_events = parsed["traceEvents"].as_array().unwrap();

    // Should have only metadata events (no divergence-related duration/counter events)
    let duration_events: Vec<_> = trace_events
        .iter()
        .filter(|e| e["ph"] == "X")
        .collect();
    assert!(
        duration_events.is_empty(),
        "Identical traces should produce no duration events, got {}",
        duration_events.len()
    );

    let counter_events: Vec<_> = trace_events
        .iter()
        .filter(|e| e["ph"] == "C")
        .collect();
    assert!(
        counter_events.is_empty(),
        "Identical traces should produce no counter events, got {}",
        counter_events.len()
    );

    // Metadata event for process name should still be present
    let metadata_events: Vec<_> = trace_events
        .iter()
        .filter(|e| e["ph"] == "M")
        .collect();
    assert_eq!(metadata_events.len(), 1, "Should have exactly one metadata event (process_name)");
}
