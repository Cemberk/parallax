//! Rendering logic for the 3-pane TUI layout.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::aligned::{event_type_short, AlignedRow, EventDisplay, RowDivergence};
use super::app::{App, FocusPane, InputMode};
use crate::differ::SnapshotContext;
use crate::site_map::SiteMap;
use crate::trace_format::HistoryEntry;

/// Which side of the split pane we're rendering.
#[derive(Clone, Copy)]
enum PaneSide {
    Left,
    Right,
}

/// Main draw function -- renders all 3 panes + status bar.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    // Vertical split: event panes | detail pane | status bar.
    let detail_height = if app.aligned.has_snapshot() {
        24  // Extra room for per-lane operand table
    } else if app.aligned.has_history() {
        18
    } else {
        10
    };
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(detail_height),
            Constraint::Length(1),
        ])
        .split(area);

    // Horizontal split for the two trace panes.
    let trace_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_chunks[0]);

    // Update terminal height for page calculations.
    app.terminal_height = area.height;

    draw_event_pane(frame, app, trace_chunks[0], PaneSide::Left);
    draw_event_pane(frame, app, trace_chunks[1], PaneSide::Right);
    draw_detail_pane(frame, app, main_chunks[1]);
    draw_status_bar(frame, app, main_chunks[2]);
}

fn draw_event_pane(frame: &mut Frame, app: &mut App, area: Rect, side: PaneSide) {
    let warp_idx = app.current_warp;
    let view = app.aligned.warp_view(warp_idx);

    let visible_height = area.height.saturating_sub(2) as usize; // minus borders
    let start = app.scroll_offset;
    let end = (start + visible_height).min(view.rows.len());

    let title = match side {
        PaneSide::Left => format!(" Trace A -- Warp {} ", warp_idx),
        PaneSide::Right => format!(" Trace B -- Warp {} ", warp_idx),
    };

    let is_focused = match side {
        PaneSide::Left => app.focus == FocusPane::TraceA,
        PaneSide::Right => app.focus == FocusPane::TraceB,
    };

    let border_style = if is_focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let lines: Vec<Line> = if view.rows.is_empty() {
        vec![Line::from(Span::styled(
            "  (no events)",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        view.rows[start..end]
            .iter()
            .map(|row| format_event_line(row, side, row.row_idx == app.selected_row, area.width))
            .collect()
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(border_style);

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn format_event_line(row: &AlignedRow, side: PaneSide, is_selected: bool, width: u16) -> Line {
    let evt = match side {
        PaneSide::Left => &row.event_a,
        PaneSide::Right => &row.event_b,
    };

    let has_divergence = !row.divergences.is_empty();

    match evt {
        Some(e) => {
            let mut spans = Vec::new();

            // Row index.
            spans.push(Span::styled(
                format!("{:>4} ", row.row_idx),
                Style::default().fg(Color::DarkGray),
            ));

            // Event type badge.
            let type_str = event_type_short(e.event_type);
            let type_color = match e.event_type {
                0 => Color::Yellow,  // BRN
                1 => Color::Magenta, // SHM
                2 => Color::Blue,    // ATM
                3 => Color::Green,   // FEN
                4 => Color::Green,   // FEX
                _ => Color::White,
            };
            spans.push(Span::styled(
                format!("{} ", type_str),
                Style::default().fg(type_color),
            ));

            // Site ID.
            spans.push(Span::styled(
                format!("{:08x} ", e.site_id),
                Style::default().fg(Color::Cyan),
            ));

            // Branch direction (for branch events).
            if e.event_type == 0 {
                let dir_str = if e.branch_dir == 0 { "NT" } else { "TK" };
                spans.push(Span::styled(
                    format!("{} ", dir_str),
                    Style::default().fg(Color::White),
                ));
            }

            // Active mask (compact).
            spans.push(Span::styled(
                format!("m:{:08x} ", e.active_mask),
                Style::default().fg(Color::DarkGray),
            ));

            // Source location (if space allows).
            if width > 50 {
                if let Some(ref loc) = e.source_loc {
                    spans.push(Span::styled(
                        loc.clone(),
                        Style::default().fg(Color::Green),
                    ));
                }
            }

            let mut style = Style::default();
            if is_selected {
                style = style.add_modifier(Modifier::REVERSED);
            }
            if has_divergence {
                // Pick color based on the most severe divergence.
                let color = divergence_color(&row.divergences);
                style = style.fg(color);
            }

            Line::styled(
                spans.iter().map(|s| s.content.as_ref()).collect::<String>(),
                Style::default(),
            );
            // Build line from styled spans, then apply row-level style.
            let mut line = Line::from(spans);
            if is_selected || has_divergence {
                line = line.style(style);
            }
            line
        }
        None => {
            // Gap row (event only on the other side).
            let style = if is_selected {
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC)
            };
            Line::from(Span::styled(
                format!("{:>4} --- (no event) ---", row.row_idx),
                style,
            ))
        }
    }
}

fn divergence_color(divs: &[RowDivergence]) -> Color {
    for d in divs {
        match d {
            RowDivergence::Path { .. } => return Color::Red,
            _ => {}
        }
    }
    for d in divs {
        match d {
            RowDivergence::Branch { .. } => return Color::Yellow,
            RowDivergence::ActiveMask { .. } => return Color::Magenta,
            RowDivergence::Value { .. } => return Color::Blue,
            RowDivergence::ExtraInA | RowDivergence::ExtraInB => return Color::Cyan,
            _ => {}
        }
    }
    Color::White
}

fn draw_detail_pane(frame: &mut Frame, app: &mut App, area: Rect) {
    let is_focused = app.focus == FocusPane::Detail;
    let border_style = if is_focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let is_identical = app.total_divergences == 0;
    let warp_idx = app.current_warp;
    let selected = app.selected_row;

    // Phase 1: get detail lines (warp_view takes &mut self for cache)
    let (mut content, has_divergences) = if is_identical {
        (
            vec![Line::from(Span::styled(
                "  Traces are IDENTICAL -- no divergences found",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ))],
            false,
        )
    } else {
        let view = app.aligned.warp_view(warp_idx);
        if let Some(row) = view.rows.get(selected) {
            let has_divs = !row.divergences.is_empty();
            (format_detail(row), has_divs)
        } else {
            (vec![Line::from("  No event selected")], false)
        }
    };

    // Phase 2: append snapshot context (per-lane operands) for branch divergences
    if has_divergences && app.aligned.has_snapshot() {
        // Find the matching divergence in diff_result to get snapshot data
        let site_id = {
            let view = app.aligned.warp_view(warp_idx);
            view.rows.get(selected).and_then(|row| {
                row.event_a.as_ref().map(|e| e.site_id)
            })
        };
        if let Some(sid) = site_id {
            let snap = app.aligned.diff_result.divergences.iter()
                .find(|d| d.warp_idx == warp_idx && d.site_id == sid && d.snapshot.is_some())
                .and_then(|d| d.snapshot.as_ref());
            if let Some(snap) = snap {
                format_snapshot_lines(&mut content, snap);
            }
        }
    }

    // Phase 3: append history (separate borrow, warp_view borrow is released)
    if has_divergences && app.aligned.has_history() {
        let (hist_a, hist_b) = app.aligned.get_history(warp_idx);
        let site_map = app.aligned.site_map.as_ref();
        format_history_lines(&mut content, &hist_a, &hist_b, site_map);
    }

    let block = Block::default()
        .title(" Detail ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let paragraph = Paragraph::new(content).block(block);
    frame.render_widget(paragraph, area);
}

fn format_detail(row: &AlignedRow) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if row.divergences.is_empty() {
        // No divergence -- show event details.
        if let Some(ref e) = row.event_a {
            lines.push(Line::from(vec![
                Span::styled("  Event: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!(
                        "{} at site 0x{:08x}",
                        event_type_long(e.event_type),
                        e.site_id
                    ),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
            if let Some(ref loc) = e.source_loc {
                lines.push(Line::from(vec![
                    Span::styled("  Source: ", Style::default().fg(Color::White)),
                    Span::styled(loc.clone(), Style::default().fg(Color::Green)),
                ]));
            }
            lines.push(Line::from(vec![
                Span::styled("  Value: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{} (0x{:08x})", e.value_a, e.value_a),
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Mask:  ", Style::default().fg(Color::White)),
                Span::styled(
                    format!(
                        "0x{:08x} ({} threads active)",
                        e.active_mask,
                        e.active_mask.count_ones()
                    ),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        return lines;
    }

    // Show divergence details.
    for div in &row.divergences {
        match div {
            RowDivergence::Branch { dir_a, dir_b } => {
                lines.push(Line::from(Span::styled(
                    "  Branch Direction Divergence",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(vec![
                    Span::styled("  Trace A: ", Style::default().fg(Color::White)),
                    Span::styled(
                        if *dir_a == 0 { "NOT-TAKEN" } else { "TAKEN" },
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("  Trace B: ", Style::default().fg(Color::White)),
                    Span::styled(
                        if *dir_b == 0 { "NOT-TAKEN" } else { "TAKEN" },
                        Style::default().fg(Color::Magenta),
                    ),
                ]));
            }
            RowDivergence::ActiveMask { mask_a, mask_b } => {
                lines.push(Line::from(Span::styled(
                    "  Active Mask Mismatch",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )));
                // 32-lane visual.
                lines.push(format_mask_line("  A: ", *mask_a, *mask_b));
                lines.push(format_mask_line("  B: ", *mask_b, *mask_a));
                let diff = mask_a ^ mask_b;
                lines.push(Line::from(vec![
                    Span::styled("  Differ: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!(
                            "0x{:08x} ({} lanes)",
                            diff,
                            diff.count_ones()
                        ),
                        Style::default().fg(Color::Red),
                    ),
                ]));
            }
            RowDivergence::Value { val_a, val_b } => {
                lines.push(Line::from(Span::styled(
                    "  Value Mismatch",
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(vec![
                    Span::styled("  Trace A: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{} (0x{:08x})", val_a, val_a),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("  Trace B: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{} (0x{:08x})", val_b, val_b),
                        Style::default().fg(Color::Magenta),
                    ),
                ]));
            }
            RowDivergence::Path { site_a, site_b } => {
                lines.push(Line::from(Span::styled(
                    "  TRUE PATH DIVERGENCE",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(vec![
                    Span::styled("  Trace A reached: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("0x{:08x}", site_a),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("  Trace B reached: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("0x{:08x}", site_b),
                        Style::default().fg(Color::Magenta),
                    ),
                ]));
                lines.push(Line::from(Span::styled(
                    "  Control flow has truly diverged",
                    Style::default().fg(Color::Red),
                )));
            }
            RowDivergence::ExtraInA => {
                lines.push(Line::from(Span::styled(
                    "  Extra Event (Drift) -- only in Trace A",
                    Style::default().fg(Color::Cyan),
                )));
            }
            RowDivergence::ExtraInB => {
                lines.push(Line::from(Span::styled(
                    "  Extra Event (Drift) -- only in Trace B",
                    Style::default().fg(Color::Cyan),
                )));
            }
        }
    }

    // Show source location if available.
    if let Some(ref e) = row.event_a {
        if let Some(ref loc) = e.source_loc {
            lines.push(Line::from(vec![
                Span::styled("  Source: ", Style::default().fg(Color::White)),
                Span::styled(loc.clone(), Style::default().fg(Color::Green)),
            ]));
        }
    }

    lines
}

/// Render a 32-lane active mask as a visual bar.
fn format_mask_line(prefix: &str, mask: u32, other: u32) -> Line<'static> {
    let diff = mask ^ other;
    let mut spans = vec![Span::styled(
        prefix.to_string(),
        Style::default().fg(Color::White),
    )];

    for lane in (0..32).rev() {
        let bit = (mask >> lane) & 1;
        let differs = (diff >> lane) & 1;

        let ch = if bit == 1 { "#" } else { "." };
        let color = if differs == 1 {
            Color::Red
        } else if bit == 1 {
            Color::Green
        } else {
            Color::DarkGray
        };

        spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
    }

    spans.push(Span::styled(
        format!(" ({})", mask.count_ones()),
        Style::default().fg(Color::DarkGray),
    ));

    Line::from(spans)
}

/// Append value history lines to the detail content.
fn format_history_lines(
    lines: &mut Vec<Line<'static>>,
    hist_a: &[HistoryEntry],
    hist_b: &[HistoryEntry],
    site_map: Option<&SiteMap>,
) {
    if hist_a.is_empty() && hist_b.is_empty() {
        return;
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Value History:",
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )));

    let show_count = 4; // Show last N entries to fit in pane

    if !hist_a.is_empty() {
        lines.push(Line::from(Span::styled(
            "    Trace A:",
            Style::default().fg(Color::Cyan),
        )));
        let start = hist_a.len().saturating_sub(show_count);
        for (i, entry) in hist_a[start..].iter().enumerate() {
            let offset = i as i32 - (hist_a[start..].len() as i32);
            let loc = site_map
                .and_then(|m| m.get(entry.site_id))
                .map(|l| l.format_short())
                .unwrap_or_else(|| format!("0x{:08x}", entry.site_id));
            lines.push(Line::from(vec![
                Span::styled(
                    format!("      [{:+2}] ", offset),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("val={:<10} ", entry.value),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(loc, Style::default().fg(Color::Green)),
            ]));
        }
    }

    if !hist_b.is_empty() {
        lines.push(Line::from(Span::styled(
            "    Trace B:",
            Style::default().fg(Color::Magenta),
        )));
        let start = hist_b.len().saturating_sub(show_count);
        for (i, entry) in hist_b[start..].iter().enumerate() {
            let offset = i as i32 - (hist_b[start..].len() as i32);
            let loc = site_map
                .and_then(|m| m.get(entry.site_id))
                .map(|l| l.format_short())
                .unwrap_or_else(|| format!("0x{:08x}", entry.site_id));
            lines.push(Line::from(vec![
                Span::styled(
                    format!("      [{:+2}] ", offset),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("val={:<10} ", entry.value),
                    Style::default().fg(Color::Magenta),
                ),
                Span::styled(loc, Style::default().fg(Color::Green)),
            ]));
        }
    }
}

/// Append per-lane operand snapshot lines to the detail content.
fn format_snapshot_lines(lines: &mut Vec<Line<'static>>, snap: &SnapshotContext) {
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Per-Lane Operands:",
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )));

    // Header row
    lines.push(Line::from(vec![
        Span::styled("    Lane  ", Style::default().fg(Color::DarkGray)),
        Span::styled("A:lhs     ", Style::default().fg(Color::Cyan)),
        Span::styled("A:rhs     ", Style::default().fg(Color::Cyan)),
        Span::styled("B:lhs     ", Style::default().fg(Color::Magenta)),
        Span::styled("B:rhs", Style::default().fg(Color::Magenta)),
    ]));

    let active_lanes = snap.mask_a | snap.mask_b;
    let mut shown = 0;
    for lane in 0..32u32 {
        if (active_lanes >> lane) & 1 == 0 {
            continue;
        }
        if shown >= 6 {
            let remaining = (lane..32)
                .filter(|l| (active_lanes >> l) & 1 != 0)
                .count();
            if remaining > 0 {
                lines.push(Line::from(Span::styled(
                    format!("    ... {} more active lanes", remaining),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            break;
        }

        let la = snap.lhs_a[lane as usize] as i32;
        let ra = snap.rhs_a[lane as usize] as i32;
        let lb = snap.lhs_b[lane as usize] as i32;
        let rb = snap.rhs_b[lane as usize] as i32;
        let differs = la != lb as i32 || ra != rb as i32;

        let color = if differs { Color::Red } else { Color::White };
        let marker = if differs { " <<<" } else { "" };

        lines.push(Line::from(vec![Span::styled(
            format!(
                "    {:>4}  {:<10}{:<10}{:<10}{}{}",
                lane, la, ra, lb, rb, marker
            ),
            Style::default().fg(color),
        )]));
        shown += 1;
    }
}

fn draw_status_bar(frame: &mut Frame, app: &mut App, area: Rect) {
    let line = match app.input_mode {
        InputMode::WarpJump => {
            Line::from(vec![
                Span::styled(" Jump to warp: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    app.jump_input.clone(),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "_  (Enter=confirm, Esc=cancel)",
                    Style::default().fg(Color::DarkGray),
                ),
            ])
        }
        InputMode::Normal => {
            let num_warps = app.aligned.num_warps();
            let view = app.aligned.warp_view(app.current_warp);
            let total_rows = view.rows.len();
            let (div_pos, div_total) = app.current_div_position();

            let kernel_name = app.aligned.trace_a.header().kernel_name_str().to_string();

            Line::from(vec![
                Span::styled(
                    format!(" Warp {}/{}", app.current_warp, num_warps),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(" | ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("Row {}/{}", app.selected_row + 1, total_rows),
                    Style::default().fg(Color::White),
                ),
                Span::styled(" | ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("Div {}/{}", div_pos, div_total),
                    if div_total > 0 {
                        Style::default().fg(Color::Red)
                    } else {
                        Style::default().fg(Color::Green)
                    },
                ),
                Span::styled(" | ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    if app.aligned.has_history() {
                        "[n]ext [N]prev [/]warp [q]uit [hist] "
                    } else {
                        "[n]ext [N]prev [/]warp [q]uit "
                    },
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(kernel_name, Style::default().fg(Color::DarkGray)),
            ])
        }
    };

    let bg = Style::default().bg(Color::DarkGray).fg(Color::White);
    let paragraph = Paragraph::new(line).style(bg);
    frame.render_widget(paragraph, area);
}

fn event_type_long(t: u8) -> &'static str {
    match t {
        0 => "Branch",
        1 => "Shared Memory Store",
        2 => "Atomic",
        3 => "Function Entry",
        4 => "Function Exit",
        5 => "Switch",
        _ => "Unknown",
    }
}
