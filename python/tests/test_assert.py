"""Tests for the ``prlx assert`` CLI command."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


class TestCmdAssertMissingTraces:
    def test_cmd_assert_missing_traces(self, capsys):
        """No trace_a_pos and no golden -> should print error."""
        from prlx.cli import cmd_assert

        args = SimpleNamespace(
            golden=None,
            trace_a_pos=None,
            trace=None,
            max_divergences=0,
            json=False,
            ignore_active_mask=False,
            session=False,
            map=None,
        )
        result = cmd_assert(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "two trace files" in captured.err or "golden" in captured.err


class TestCmdAssertGoldenMode:
    def test_cmd_assert_golden_mode(self, tmp_path):
        """--golden sets trace_a correctly; positional becomes trace_b."""
        from prlx.cli import cmd_assert

        golden_file = tmp_path / "golden.prlx"
        test_file = tmp_path / "test.prlx"
        golden_file.write_bytes(b"\x00" * 160)
        test_file.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            golden=str(golden_file),
            trace_a_pos=None,
            trace=str(test_file),
            max_divergences=0,
            json=False,
            ignore_active_mask=False,
            session=False,
            map=None,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout='{"total_divergences": 0}', stderr=""
                )
                result = cmd_assert(args)

                # Verify the command was called with golden as trace_a
                call_args = mock_run.call_args[0][0]
                assert str(golden_file) in call_args
                assert str(test_file) in call_args
                # golden should come before test in the command
                golden_idx = call_args.index(str(golden_file))
                test_idx = call_args.index(str(test_file))
                assert golden_idx < test_idx


class TestCmdAssertJsonPassthrough:
    def test_cmd_assert_json_passthrough(self, tmp_path, capsys):
        """Verify --json flag causes raw JSON passthrough to stdout."""
        from prlx.cli import cmd_assert

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            golden=None,
            trace_a_pos=str(trace_a),
            trace=str(trace_b),
            max_divergences=0,
            json=True,
            ignore_active_mask=False,
            session=False,
            map=None,
        )

        json_output = '{"total_divergences": 3, "divergences": []}'

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json_output, stderr=""
                )
                result = cmd_assert(args)

                # Verify --json is passed to subprocess command
                call_args = mock_run.call_args[0][0]
                assert "--json" in call_args

                # Verify raw JSON is written to stdout
                captured = capsys.readouterr()
                assert json_output in captured.out


class TestCmdAssertThresholdLogic:
    def test_cmd_assert_threshold_logic(self, tmp_path):
        """Verify --max-divergences is passed correctly to subprocess."""
        from prlx.cli import cmd_assert

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            golden=None,
            trace_a_pos=str(trace_a),
            trace=str(trace_b),
            max_divergences=5,
            json=False,
            ignore_active_mask=False,
            session=False,
            map=None,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout='{"total_divergences": 3}', stderr=""
                )
                result = cmd_assert(args)

                call_args = mock_run.call_args[0][0]
                assert "--max-allowed-divergences" in call_args
                idx = call_args.index("--max-allowed-divergences")
                assert call_args[idx + 1] == "5"

    def test_cmd_assert_zero_threshold_omits_flag(self, tmp_path):
        """When --max-divergences is 0, the flag should NOT be passed to prlx-diff."""
        from prlx.cli import cmd_assert

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            golden=None,
            trace_a_pos=str(trace_a),
            trace=str(trace_b),
            max_divergences=0,
            json=False,
            ignore_active_mask=False,
            session=False,
            map=None,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout='{"total_divergences": 0}', stderr=""
                )
                result = cmd_assert(args)

                call_args = mock_run.call_args[0][0]
                assert "--max-allowed-divergences" not in call_args


class TestCmdAssertIgnoreActiveMask:
    def test_cmd_assert_ignore_active_mask(self, tmp_path):
        """Verify --ignore-active-mask flag is forwarded."""
        from prlx.cli import cmd_assert

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            golden=None,
            trace_a_pos=str(trace_a),
            trace=str(trace_b),
            max_divergences=0,
            json=False,
            ignore_active_mask=True,
            session=False,
            map=None,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = "/fake/prlx-diff"
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0,
                    stdout='{"total_divergences": 0}', stderr=""
                )
                cmd_assert(args)

                call_args = mock_run.call_args[0][0]
                assert "--ignore-active-mask" in call_args


class TestCmdAssertDifferNotFound:
    def test_cmd_assert_differ_not_found(self, tmp_path, capsys):
        """When prlx-diff binary is not found, should print error."""
        from prlx.cli import cmd_assert

        trace_a = tmp_path / "a.prlx"
        trace_b = tmp_path / "b.prlx"
        trace_a.write_bytes(b"\x00" * 160)
        trace_b.write_bytes(b"\x00" * 160)

        args = SimpleNamespace(
            golden=None,
            trace_a_pos=str(trace_a),
            trace=str(trace_b),
            max_divergences=0,
            json=False,
            ignore_active_mask=False,
            session=False,
            map=None,
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_differ_binary.return_value = None
            result = cmd_assert(args)
            assert result == 1
            captured = capsys.readouterr()
            assert "prlx-diff" in captured.err
