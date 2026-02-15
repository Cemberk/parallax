//! Application state and navigation logic for the TUI viewer.

use crate::differ::DiffResult;
use crate::parser::TraceFile;
use crate::site_map::SiteMap;

use super::aligned::AlignedTrace;
use super::input::Action;
use super::source_cache::SourceCache;

/// Which pane has keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPane {
    TraceA,
    TraceB,
    Detail,
}

/// Current input mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    WarpJump,
}

/// Per-warp summary for fast overview.
#[derive(Debug, Clone)]
pub struct WarpSummary {
    pub warp_idx: u32,
    pub num_divergences: usize,
}

pub struct App {
    pub aligned: AlignedTrace,

    // Navigation
    pub current_warp: u32,
    pub scroll_offset: usize,
    pub selected_row: usize,
    pub focus: FocusPane,
    pub input_mode: InputMode,
    pub jump_input: String,

    // Pre-computed
    pub warp_summaries: Vec<WarpSummary>,
    /// Indices into warp_summaries for warps that have divergences.
    pub divergent_warp_indices: Vec<usize>,
    pub total_divergences: usize,

    pub float_format: bool,
    pub quit: bool,
    pub terminal_height: u16,

    // Source view
    pub source_view_enabled: bool,
    pub source_cache: SourceCache,
}

impl App {
    pub fn new(
        trace_a: TraceFile,
        trace_b: TraceFile,
        diff_result: DiffResult,
        site_map: Option<SiteMap>,
        float_format: bool,
    ) -> Self {
        let num_warps = diff_result.total_warps;
        let total_divergences = diff_result.divergences.len();

        let mut div_counts: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for div in &diff_result.divergences {
            *div_counts.entry(div.warp_idx).or_default() += 1;
        }

        let mut warp_summaries = Vec::with_capacity(num_warps);
        let mut divergent_warp_indices = Vec::new();
        for i in 0..num_warps as u32 {
            let n = div_counts.get(&i).copied().unwrap_or(0);
            if n > 0 {
                divergent_warp_indices.push(warp_summaries.len());
            }
            warp_summaries.push(WarpSummary {
                warp_idx: i,
                num_divergences: n,
            });
        }

        let start_warp = divergent_warp_indices
            .first()
            .map(|&idx| warp_summaries[idx].warp_idx)
            .unwrap_or(0);

        let aligned = AlignedTrace::new(trace_a, trace_b, diff_result, site_map);

        App {
            aligned,
            current_warp: start_warp,
            scroll_offset: 0,
            selected_row: 0,
            focus: FocusPane::TraceA,
            input_mode: InputMode::Normal,
            jump_input: String::new(),
            warp_summaries,
            divergent_warp_indices,
            total_divergences,
            float_format,
            quit: false,
            terminal_height: 24,
            source_view_enabled: false,
            source_cache: SourceCache::new(),
        }
    }

    /// Handle an action (mapped from a key event).
    pub fn handle_action(&mut self, action: Action) {
        match action {
            Action::Quit => self.quit = true,
            Action::ScrollDown(n) => self.scroll_down(n),
            Action::ScrollUp(n) => self.scroll_up(n),
            Action::PageDown => {
                let page = self.visible_rows() / 2;
                self.scroll_down(page.max(1));
            }
            Action::PageUp => {
                let page = self.visible_rows() / 2;
                self.scroll_up(page.max(1));
            }
            Action::GoTop => {
                self.selected_row = 0;
                self.scroll_offset = 0;
            }
            Action::GoBottom => {
                let view = self.aligned.warp_view(self.current_warp);
                if !view.rows.is_empty() {
                    self.selected_row = view.rows.len() - 1;
                    self.clamp_scroll();
                }
            }
            Action::NextDivergence => self.next_divergence(),
            Action::PrevDivergence => self.prev_divergence(),
            Action::CycleFocus => self.cycle_focus(),
            Action::StartWarpJump => {
                self.input_mode = InputMode::WarpJump;
                self.jump_input.clear();
            }
            Action::WarpJumpChar(c) => {
                self.jump_input.push(c);
            }
            Action::WarpJumpConfirm => {
                if let Ok(idx) = self.jump_input.parse::<u32>() {
                    self.jump_to_warp(idx);
                }
                self.input_mode = InputMode::Normal;
                self.jump_input.clear();
            }
            Action::WarpJumpCancel => {
                self.input_mode = InputMode::Normal;
                self.jump_input.clear();
            }
            Action::NextWarp => {
                if self.current_warp + 1 < self.aligned.num_warps() {
                    self.jump_to_warp(self.current_warp + 1);
                }
            }
            Action::PrevWarp => {
                if self.current_warp > 0 {
                    self.jump_to_warp(self.current_warp - 1);
                }
            }
            Action::ToggleSource => {
                self.source_view_enabled = !self.source_view_enabled;
            }
            Action::None => {}
        }
    }

    fn visible_rows(&self) -> usize {
        // Event pane takes ~70% of terminal minus borders/status.
        // Approximate: terminal_height - detail(12) - status(1) - borders(2).
        self.terminal_height.saturating_sub(15) as usize
    }

    fn scroll_down(&mut self, n: usize) {
        let view = self.aligned.warp_view(self.current_warp);
        let max_row = view.rows.len().saturating_sub(1);
        self.selected_row = (self.selected_row + n).min(max_row);
        self.clamp_scroll();
    }

    fn scroll_up(&mut self, n: usize) {
        self.selected_row = self.selected_row.saturating_sub(n);
        self.clamp_scroll();
    }

    fn clamp_scroll(&mut self) {
        let vis = self.visible_rows();
        if vis == 0 {
            return;
        }
        // Ensure selected_row is within [scroll_offset, scroll_offset + vis).
        if self.selected_row < self.scroll_offset {
            self.scroll_offset = self.selected_row;
        } else if self.selected_row >= self.scroll_offset + vis {
            self.scroll_offset = self.selected_row - vis + 1;
        }
    }

    fn next_divergence(&mut self) {
        let view = self.aligned.warp_view(self.current_warp);
        let next_in_warp = view
            .divergence_indices
            .iter()
            .find(|&&idx| idx > self.selected_row);

        if let Some(&idx) = next_in_warp {
            self.selected_row = idx;
            self.clamp_scroll();
        } else {
            let cur_warp_pos = self
                .divergent_warp_indices
                .iter()
                .position(|&i| self.warp_summaries[i].warp_idx == self.current_warp);

            let next_warp_pos = match cur_warp_pos {
                Some(pos) if pos + 1 < self.divergent_warp_indices.len() => Some(pos + 1),
                None => {
                    // Current warp has no divergences; find next one after it.
                    self.divergent_warp_indices
                        .iter()
                        .position(|&i| self.warp_summaries[i].warp_idx > self.current_warp)
                }
                _ => None,
            };

            if let Some(pos) = next_warp_pos {
                let warp_idx = self.warp_summaries[self.divergent_warp_indices[pos]].warp_idx;
                self.current_warp = warp_idx;
                self.scroll_offset = 0;
                self.selected_row = 0;
                let view = self.aligned.warp_view(warp_idx);
                if let Some(&first) = view.divergence_indices.first() {
                    self.selected_row = first;
                    self.clamp_scroll();
                }
            }
        }
    }

    fn prev_divergence(&mut self) {
        let view = self.aligned.warp_view(self.current_warp);
        let prev_in_warp = view
            .divergence_indices
            .iter()
            .rev()
            .find(|&&idx| idx < self.selected_row);

        if let Some(&idx) = prev_in_warp {
            self.selected_row = idx;
            self.clamp_scroll();
        } else {
            let cur_warp_pos = self
                .divergent_warp_indices
                .iter()
                .position(|&i| self.warp_summaries[i].warp_idx == self.current_warp);

            let prev_warp_pos = match cur_warp_pos {
                Some(pos) if pos > 0 => Some(pos - 1),
                None => {
                    self.divergent_warp_indices
                        .iter()
                        .rposition(|&i| self.warp_summaries[i].warp_idx < self.current_warp)
                }
                _ => None,
            };

            if let Some(pos) = prev_warp_pos {
                let warp_idx = self.warp_summaries[self.divergent_warp_indices[pos]].warp_idx;
                self.current_warp = warp_idx;
                self.scroll_offset = 0;
                self.selected_row = 0;
                let view = self.aligned.warp_view(warp_idx);
                if let Some(&last) = view.divergence_indices.last() {
                    self.selected_row = last;
                    self.clamp_scroll();
                }
            }
        }
    }

    fn jump_to_warp(&mut self, idx: u32) {
        if idx < self.aligned.num_warps() {
            self.current_warp = idx;
            self.scroll_offset = 0;
            self.selected_row = 0;
        }
    }

    fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            FocusPane::TraceA => FocusPane::TraceB,
            FocusPane::TraceB => FocusPane::Detail,
            FocusPane::Detail => FocusPane::TraceA,
        };
    }

    /// Count which divergence number the current selection is (1-based), or 0.
    pub fn current_div_position(&mut self) -> (usize, usize) {
        let mut global_pos = 0usize;
        for ws in &self.warp_summaries {
            if ws.warp_idx < self.current_warp {
                global_pos += ws.num_divergences;
            } else if ws.warp_idx == self.current_warp {
                let view = self.aligned.warp_view(ws.warp_idx);
                let local = view
                    .divergence_indices
                    .iter()
                    .position(|&idx| idx >= self.selected_row)
                    .unwrap_or(view.divergence_indices.len());
                // If we are exactly on a divergence, count it.
                let on_div = view.divergence_indices.contains(&self.selected_row);
                global_pos += if on_div { local + 1 } else { local };
                break;
            }
        }
        (global_pos, self.total_divergences)
    }
}
