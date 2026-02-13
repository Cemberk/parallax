//! Integration tests for the differential analysis engine

use gddbg_diff::differ::{diff_traces, DiffConfig, DivergenceKind};
use gddbg_diff::trace_format::*;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper: Create a minimal valid trace file with given events
fn create_test_trace(kernel_name: &str, events: &[Vec<TraceEvent>]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    // Create header
    let mut header = TraceFileHeader {
        magic: GDDBG_MAGIC,
        version: GDDBG_VERSION,
        flags: 0,
        kernel_name_hash: 0x1234567890ABCDEF,
        kernel_name: [0; 64],
        grid_dim: [1, 1, 1],
        block_dim: [32, 1, 1],
        num_warps_per_block: 1,
        total_warp_slots: events.len() as u32,
        events_per_warp: GDDBG_EVENTS_PER_WARP as u32,
        _pad: 0,
        timestamp: 1234567890,
        cuda_arch: 80,
        _reserved: [0; 5],
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
            _reserved: 0,
        };
        let warp_header_bytes: &[u8] = bytemuck::bytes_of(&warp_header);
        file.write_all(warp_header_bytes).unwrap();

        // Write events
        for event in warp_events {
            let event_bytes: &[u8] = bytemuck::bytes_of(event);
            file.write_all(event_bytes).unwrap();
        }

        // Pad remaining events with zeros
        let remaining = GDDBG_EVENTS_PER_WARP - warp_events.len();
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

    let trace_a = gddbg_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = gddbg_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

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

    let trace_a = gddbg_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = gddbg_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

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

    let trace_a = gddbg_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = gddbg_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

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

    let trace_a = gddbg_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = gddbg_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

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

    let trace_a = gddbg_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = gddbg_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

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

    let trace_a = gddbg_diff::parser::TraceFile::open(trace_a_file.path()).unwrap();
    let trace_b = gddbg_diff::parser::TraceFile::open(trace_b_file.path()).unwrap();

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
