"""Tests for prlx.cli."""

import argparse
from pathlib import Path

import pytest


class TestDiffArgs:
    def _make_parser(self):
        import importlib
        import prlx.cli
        importlib.reload(prlx.cli)

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        p_diff = subparsers.add_parser("diff")
        p_diff.add_argument("trace_a")
        p_diff.add_argument("trace_b")
        p_diff.add_argument("--values", action="store_true")
        p_diff.add_argument("-n", "--max-shown", type=int, default=50)
        p_diff.add_argument("--lookahead", type=int, default=32)
        p_diff.add_argument("--tui", action="store_true")
        p_diff.add_argument("--force", action="store_true")
        p_diff.add_argument("--map", dest="site_map")
        return parser

    def test_force_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["diff", "a.prlx", "b.prlx", "--force"])
        assert args.force is True
        assert args.command == "diff"

    def test_values_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["diff", "a.prlx", "b.prlx", "--values"])
        assert args.values is True

    def test_max_shown(self):
        parser = self._make_parser()
        args = parser.parse_args(["diff", "a.prlx", "b.prlx", "-n", "5"])
        assert args.max_shown == 5

    def test_default_values(self):
        parser = self._make_parser()
        args = parser.parse_args(["diff", "a.prlx", "b.prlx"])
        assert args.force is False
        assert args.values is False
        assert args.max_shown == 50
        assert args.lookahead == 32
        assert args.tui is False

    def test_all_flags_combined(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "diff", "a.prlx", "b.prlx",
            "--force", "--values", "--tui", "-n", "10", "--lookahead", "64",
        ])
        assert args.force is True
        assert args.values is True
        assert args.tui is True
        assert args.max_shown == 10
        assert args.lookahead == 64


class TestFindSiteMap:
    def test_find_site_map_in_directory(self, tmp_path):
        from prlx.cli import find_site_map
        site_map = tmp_path / "prlx-sites.json"
        site_map.write_text("{}")
        trace_file = tmp_path / "trace.prlx"
        trace_file.write_bytes(b"\x00")

        result = find_site_map(str(trace_file))
        assert result is not None
        assert Path(result).name == "prlx-sites.json"

    def test_find_site_map_not_found(self, tmp_path):
        from prlx.cli import find_site_map
        trace_file = tmp_path / "trace.prlx"
        trace_file.write_bytes(b"\x00")

        result = find_site_map(str(trace_file))
        assert result is None
