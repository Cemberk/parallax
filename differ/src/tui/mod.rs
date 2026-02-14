//! Interactive TUI viewer for prlx trace comparison.
//!
//! Provides a split-pane vimdiff-like interface for navigating divergences
//! between two CUDA execution traces.

mod aligned;
mod app;
mod input;
mod render;

use std::io;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::differ::DiffResult;
use crate::parser::TraceFile;
use crate::site_map::SiteMap;

use app::App;
use input::map_key;

/// Launch the interactive TUI viewer.
pub fn run_tui(
    trace_a: TraceFile,
    trace_b: TraceFile,
    diff_result: DiffResult,
    site_map: Option<SiteMap>,
) -> Result<()> {
    // Terminal setup.
    enable_raw_mode().context("Failed to enable raw mode. Is this a TTY?")?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(trace_a, trace_b, diff_result, site_map);

    let result = run_event_loop(&mut terminal, &mut app);

    // Terminal cleanup (always runs).
    let _ = disable_raw_mode();
    let _ = terminal.backend_mut().execute(LeaveAlternateScreen);
    let _ = terminal.show_cursor();

    result
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|frame| render::draw(frame, app))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    let action = map_key(key, &app.input_mode);
                    app.handle_action(action);
                }
                Event::Resize(_w, h) => {
                    app.terminal_height = h;
                }
                _ => {}
            }
        }

        if app.quit {
            break;
        }
    }

    Ok(())
}
