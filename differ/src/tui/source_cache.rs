//! Source file cache for displaying code snippets in the TUI detail pane.
//!
//! Caches file contents and extracts context windows around specific lines,
//! enabling inline source display when inspecting divergences.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A snippet of source code with line numbers around a target line.
pub struct SourceSnippet {
    /// Lines of source code as (1-based line number, content) pairs.
    pub lines: Vec<(usize, String)>,
    /// The 1-based line number of the target (divergence) line.
    pub target_line: usize,
    /// The filename this snippet was loaded from.
    pub filename: String,
    /// The column number (0 if unknown).
    pub column: u32,
}

/// Caches loaded source files to avoid repeated disk access.
///
/// Files that cannot be read are cached as `None` so we don't
/// keep retrying on every frame.
pub struct SourceCache {
    files: HashMap<String, Option<Vec<String>>>,
}

impl SourceCache {
    pub fn new() -> Self {
        SourceCache {
            files: HashMap::new(),
        }
    }

    /// Get a snippet of source lines around `target_line` with `context` lines
    /// of surrounding context on each side.
    ///
    /// Returns `None` if the file cannot be read or the target line is out of range.
    pub fn get_snippet(
        &mut self,
        filename: &str,
        target_line: u32,
        context: usize,
    ) -> Option<SourceSnippet> {
        if target_line == 0 {
            return None;
        }

        let file_lines = self.load_file(filename)?;
        let total_lines = file_lines.len();

        if total_lines == 0 {
            return None;
        }

        let target = target_line as usize;
        if target > total_lines {
            return None;
        }

        let start = target.saturating_sub(context).max(1);
        let end = (target + context).min(total_lines);

        let lines: Vec<(usize, String)> = (start..=end)
            .map(|line_no| (line_no, file_lines[line_no - 1].clone()))
            .collect();

        Some(SourceSnippet {
            lines,
            target_line: target,
            filename: filename.to_string(),
            column: 0,
        })
    }

    /// Load a file's contents, caching the result.
    ///
    /// Tries the path as-is (absolute or relative to CWD) first.
    /// Returns `None` for unreadable files, and caches that result
    /// to avoid repeated disk access.
    fn load_file(&mut self, filename: &str) -> Option<&Vec<String>> {
        if !self.files.contains_key(filename) {
            let contents = self.try_read_file(filename);
            self.files.insert(filename.to_string(), contents);
        }
        self.files.get(filename).and_then(|opt| opt.as_ref())
    }

    fn try_read_file(&self, filename: &str) -> Option<Vec<String>> {
        // Try absolute path first
        let path = Path::new(filename);
        if path.is_absolute() {
            if let Ok(content) = fs::read_to_string(path) {
                return Some(content.lines().map(|l| l.to_string()).collect());
            }
        }

        // Try relative to CWD
        if let Ok(content) = fs::read_to_string(filename) {
            return Some(content.lines().map(|l| l.to_string()).collect());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_temp_file(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_basic_snippet() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap();

        let mut cache = SourceCache::new();
        let snippet = cache.get_snippet(path, 5, 2).unwrap();

        assert_eq!(snippet.target_line, 5);
        assert_eq!(snippet.lines.len(), 5); // lines 3,4,5,6,7
        assert_eq!(snippet.lines[0], (3, "line 3".to_string()));
        assert_eq!(snippet.lines[2], (5, "line 5".to_string()));
        assert_eq!(snippet.lines[4], (7, "line 7".to_string()));
    }

    #[test]
    fn test_snippet_at_start() {
        let content = "first\nsecond\nthird\nfourth\nfifth\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap();

        let mut cache = SourceCache::new();
        let snippet = cache.get_snippet(path, 1, 2).unwrap();

        assert_eq!(snippet.target_line, 1);
        assert_eq!(snippet.lines[0].0, 1);
        assert_eq!(snippet.lines[0].1, "first");
        // Should include lines 1..=3
        assert_eq!(snippet.lines.len(), 3);
    }

    #[test]
    fn test_snippet_at_end() {
        let content = "a\nb\nc\nd\ne\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap();

        let mut cache = SourceCache::new();
        let snippet = cache.get_snippet(path, 5, 2).unwrap();

        assert_eq!(snippet.target_line, 5);
        // Should include lines 3..=5
        assert_eq!(snippet.lines.len(), 3);
        assert_eq!(snippet.lines.last().unwrap().0, 5);
    }

    #[test]
    fn test_nonexistent_file_returns_none() {
        let mut cache = SourceCache::new();
        let result = cache.get_snippet("/nonexistent/path/file.cu", 10, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_caches_none_for_missing_files() {
        let mut cache = SourceCache::new();

        // First access
        let result = cache.get_snippet("/nonexistent/file.cu", 1, 5);
        assert!(result.is_none());

        // Should be cached as None now
        assert!(cache.files.contains_key("/nonexistent/file.cu"));
        assert!(cache.files.get("/nonexistent/file.cu").unwrap().is_none());

        // Second access should also return None (from cache)
        let result = cache.get_snippet("/nonexistent/file.cu", 1, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_zero_target_line_returns_none() {
        let content = "line 1\nline 2\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap();

        let mut cache = SourceCache::new();
        let result = cache.get_snippet(path, 0, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_target_line_beyond_file_returns_none() {
        let content = "only one line\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap();

        let mut cache = SourceCache::new();
        let result = cache.get_snippet(path, 100, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_file_caching() {
        let content = "cached\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap().to_string();

        let mut cache = SourceCache::new();

        // First access loads the file
        let snippet = cache.get_snippet(&path, 1, 0).unwrap();
        assert_eq!(snippet.lines[0].1, "cached");

        // Verify it is now in the cache
        assert!(cache.files.contains_key(&path));

        // Second access uses cache
        let snippet = cache.get_snippet(&path, 1, 0).unwrap();
        assert_eq!(snippet.lines[0].1, "cached");
    }

    #[test]
    fn test_snippet_with_context_larger_than_file() {
        let content = "a\nb\nc\n";
        let f = make_temp_file(content);
        let path = f.path().to_str().unwrap();

        let mut cache = SourceCache::new();
        let snippet = cache.get_snippet(path, 2, 100).unwrap();

        // Should return all lines: 1,2,3
        assert_eq!(snippet.lines.len(), 3);
        assert_eq!(snippet.target_line, 2);
    }
}
