"""Tests for the ``prlx flamegraph`` CLI command."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


class TestCmdFlamegraphPassesFlags:
    def test_cmd_flamegraph_passes_flags(self, tmp_path):
        """Verify all flags are forwarded to prlx-diff."""
        from prlx.cli import cmd_flamegraph

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        site_map = tmp_path / "prlx-sites.json"
        site_map.write_text("{}")

        args = SimpleNamespace(
            trace_a=str(trace_a),
            trace_b=str(trace_b),
            output="custom-output.json",
            map=str(site_map),
            values=True,
            force=True,
            session=True,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.call") as mock_call:
                mock_call.return_value = 0
                result = cmd_flamegraph(args)

                call_args = mock_call.call_args[0][0]

                # Verify differ binary is first argument
                assert call_args[0] == "/fake/prlx-diff"

                # Verify --export-flamegraph with output path
                assert "--export-flamegraph" in call_args
                fg_idx = call_args.index("--export-flamegraph")
                assert call_args[fg_idx + 1] == "custom-output.json"

                # Verify trace files are present
                assert str(trace_a) in call_args
                assert str(trace_b) in call_args

                # Verify all optional flags
                assert "--map" in call_args
                map_idx = call_args.index("--map")
                assert call_args[map_idx + 1] == str(site_map)

                assert "--values" in call_args
                assert "--force" in call_args
                assert "--session" in call_args

    def test_cmd_flamegraph_no_optional_flags(self, tmp_path):
        """Verify optional flags are omitted when not set."""
        from prlx.cli import cmd_flamegraph

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            trace_a=str(trace_a),
            trace_b=str(trace_b),
            output=None,
            map=None,
            values=False,
            force=False,
            session=False,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.call") as mock_call:
                mock_call.return_value = 0
                cmd_flamegraph(args)

                call_args = mock_call.call_args[0][0]
                assert "--values" not in call_args
                assert "--force" not in call_args
                assert "--session" not in call_args
                assert "--map" not in call_args


class TestCmdFlamegraphDefaultOutput:
    def test_cmd_flamegraph_default_output(self, tmp_path):
        """When no -o, should use prlx-flamegraph.json as default output."""
        from prlx.cli import cmd_flamegraph

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            trace_a=str(trace_a),
            trace_b=str(trace_b),
            output=None,  # No output specified
            map=None,
            values=False,
            force=False,
            session=False,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.call") as mock_call:
                mock_call.return_value = 0
                cmd_flamegraph(args)

                call_args = mock_call.call_args[0][0]
                assert "--export-flamegraph" in call_args
                fg_idx = call_args.index("--export-flamegraph")
                assert call_args[fg_idx + 1] == "prlx-flamegraph.json"


class TestCmdFlamegraphTraceNotFound:
    def test_cmd_flamegraph_trace_a_not_found(self, tmp_path, capsys):
        """Should error when trace A does not exist."""
        from prlx.cli import cmd_flamegraph

        trace_b = tmp_path / "b.prlx"
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            trace_a=str(tmp_path / "nonexistent.prlx"),
            trace_b=str(trace_b),
            output=None,
            map=None,
            values=False,
            force=False,
            session=False,
        )

        result = cmd_flamegraph(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Trace A not found" in captured.err

    def test_cmd_flamegraph_trace_b_not_found(self, tmp_path, capsys):
        """Should error when trace B does not exist."""
        from prlx.cli import cmd_flamegraph

        trace_a = tmp_path / "a.prlx"
        trace_a.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            trace_a=str(trace_a),
            trace_b=str(tmp_path / "nonexistent.prlx"),
            output=None,
            map=None,
            values=False,
            force=False,
            session=False,
        )

        result = cmd_flamegraph(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Trace B not found" in captured.err


class TestCmdFlamegraphDifferNotFound:
    def test_cmd_flamegraph_differ_not_found(self, tmp_path, capsys):
        """When prlx-diff binary is not found, should print error."""
        from prlx.cli import cmd_flamegraph

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            trace_a=str(trace_a),
            trace_b=str(trace_b),
            output=None,
            map=None,
            values=False,
            force=False,
            session=False,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = None
            result = cmd_flamegraph(args)
            assert result == 1
            captured = capsys.readouterr()
            assert "prlx-diff" in captured.err
