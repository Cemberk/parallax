//! Key event mapping for the TUI viewer.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::app::InputMode;

/// Actions the TUI can perform.
#[derive(Debug, Clone)]
pub enum Action {
    Quit,
    ScrollDown(usize),
    ScrollUp(usize),
    PageDown,
    PageUp,
    GoTop,
    GoBottom,
    NextDivergence,
    PrevDivergence,
    CycleFocus,
    NextWarp,
    PrevWarp,
    StartWarpJump,
    WarpJumpChar(char),
    WarpJumpConfirm,
    WarpJumpCancel,
    ToggleSource,
    None,
}

/// Map a crossterm key event to an Action.
pub fn map_key(key: KeyEvent, mode: &InputMode) -> Action {
    match mode {
        InputMode::WarpJump => map_warp_jump_key(key),
        InputMode::Normal => map_normal_key(key),
    }
}

fn map_normal_key(key: KeyEvent) -> Action {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') => Action::Quit,
            KeyCode::Char('d') => Action::PageDown,
            KeyCode::Char('u') => Action::PageUp,
            _ => Action::None,
        };
    }

    match key.code {
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Char('j') | KeyCode::Down => Action::ScrollDown(1),
        KeyCode::Char('k') | KeyCode::Up => Action::ScrollUp(1),
        KeyCode::PageDown => Action::PageDown,
        KeyCode::PageUp => Action::PageUp,
        KeyCode::Char('g') | KeyCode::Home => Action::GoTop,
        KeyCode::Char('G') | KeyCode::End => Action::GoBottom,
        KeyCode::Char('n') => Action::NextDivergence,
        KeyCode::Char('N') => Action::PrevDivergence,
        KeyCode::Tab => Action::CycleFocus,
        KeyCode::Char('/') => Action::StartWarpJump,
        KeyCode::Char(']') => Action::NextWarp,
        KeyCode::Char('[') => Action::PrevWarp,
        KeyCode::Char('s') => Action::ToggleSource,
        _ => Action::None,
    }
}

fn map_warp_jump_key(key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Enter => Action::WarpJumpConfirm,
        KeyCode::Esc => Action::WarpJumpCancel,
        KeyCode::Char(c) if c.is_ascii_digit() => Action::WarpJumpChar(c),
        KeyCode::Backspace => {
            // We'll handle backspace as cancel + restart for simplicity.
            Action::WarpJumpCancel
        }
        _ => Action::None,
    }
}
