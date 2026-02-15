"""Tests for PrlxTrainingMonitor (training loop integration).

All tests are pure mock tests -- NO GPU, NO torch import required.
"""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from prlx.training_monitor import (
    AnomalyRecord,
    AnomalyType,
    PrlxTrainingMonitor,
    StepContext,
    _CaptureState,
)


class TestAnomalyRecord:
    def test_to_dict(self):
        record = AnomalyRecord(
            step=42,
            anomaly_type=AnomalyType.NAN_INF,
            value=float("nan"),
            threshold=0.0,
            moving_average=1.5,
            timestamp=1000.0,
        )
        d = record.to_dict()
        assert d["step"] == 42
        assert d["type"] == "nan_inf"
        assert math.isnan(d["value"])
        assert d["threshold"] == 0.0
        assert d["moving_average"] == 1.5
        assert d["timestamp"] == 1000.0

    def test_loss_spike_to_dict(self):
        record = AnomalyRecord(
            step=10,
            anomaly_type=AnomalyType.LOSS_SPIKE,
            value=5.0,
            threshold=4.0,
            moving_average=2.0,
        )
        d = record.to_dict()
        assert d["type"] == "loss_spike"
        assert d["value"] == 5.0

    def test_grad_norm_to_dict(self):
        record = AnomalyRecord(
            step=20,
            anomaly_type=AnomalyType.GRAD_NORM_SPIKE,
            value=200.0,
            threshold=100.0,
            moving_average=1.0,
        )
        d = record.to_dict()
        assert d["type"] == "grad_norm_spike"


class TestMonitorInit:
    def test_default_params(self, tmp_path):
        monitor = PrlxTrainingMonitor(output_dir=str(tmp_path / "traces"))
        assert monitor.loss_threshold == 2.0
        assert monitor.nan_detection is True
        assert monitor.grad_norm_threshold == 100.0
        assert monitor.window_size == 50
        assert monitor.capture_steps_after == 3
        assert monitor.ring_buffer_size == 0
        assert monitor.max_captures == 10
        assert (tmp_path / "traces").is_dir()

    def test_custom_params(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path / "out"),
            loss_threshold=5.0,
            nan_detection=False,
            grad_norm_threshold=50.0,
            window_size=10,
            capture_steps_after=5,
            ring_buffer_size=3,
            max_captures=2,
        )
        assert monitor.loss_threshold == 5.0
        assert monitor.nan_detection is False
        assert monitor.grad_norm_threshold == 50.0
        assert monitor.window_size == 10
        assert monitor.capture_steps_after == 5
        assert monitor.ring_buffer_size == 3
        assert monitor.max_captures == 2

    def test_from_env(self, tmp_path):
        env = {
            "PRLX_TRACE_DIR": str(tmp_path / "env_traces"),
            "PRLX_LOSS_THRESHOLD": "3.5",
            "PRLX_GRAD_NORM_THRESHOLD": "50.0",
            "PRLX_WINDOW_SIZE": "20",
            "PRLX_CAPTURE_STEPS_AFTER": "5",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            monitor = PrlxTrainingMonitor.from_env()
            assert monitor.loss_threshold == 3.5
            assert monitor.grad_norm_threshold == 50.0
            assert monitor.window_size == 20
            assert monitor.capture_steps_after == 5
            assert str(monitor.output_dir) == str(tmp_path / "env_traces")


class TestNanDetection:
    def test_nan_loss_triggers_anomaly(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
        )
        monitor._record_loss(0, float("nan"))
        assert len(monitor._anomalies) == 1
        assert monitor._anomalies[0].anomaly_type == AnomalyType.NAN_INF

    def test_inf_loss_triggers_anomaly(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
        )
        monitor._record_loss(0, float("inf"))
        assert len(monitor._anomalies) == 1
        assert monitor._anomalies[0].anomaly_type == AnomalyType.NAN_INF

    def test_nan_detection_disabled(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=False,
        )
        monitor._record_loss(0, float("nan"))
        assert len(monitor._anomalies) == 0

    def test_normal_loss_no_anomaly(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
        )
        monitor._record_loss(0, 1.0)
        assert len(monitor._anomalies) == 0


class TestLossSpikeDetection:
    def test_loss_spike_after_warmup(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            loss_threshold=2.0,
            window_size=5,
        )
        # Fill warmup window
        for i in range(5):
            monitor._record_loss(i, 1.0)

        assert len(monitor._anomalies) == 0

        # Spike: 3.0 > 2.0 * 1.0
        monitor._record_loss(5, 3.0)
        assert len(monitor._anomalies) == 1
        assert monitor._anomalies[0].anomaly_type == AnomalyType.LOSS_SPIKE
        assert monitor._anomalies[0].value == 3.0

    def test_no_spike_within_threshold(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            loss_threshold=2.0,
            window_size=5,
        )
        for i in range(5):
            monitor._record_loss(i, 1.0)

        # 1.5 < 2.0 * 1.0 = 2.0
        monitor._record_loss(5, 1.5)
        assert len(monitor._anomalies) == 0

    def test_no_spike_before_warmup(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            loss_threshold=2.0,
            window_size=10,
        )
        # Only 3 steps, warmup needs 10
        for i in range(3):
            monitor._record_loss(i, 1.0)
        monitor._record_loss(3, 100.0)
        assert len(monitor._anomalies) == 0


class TestGradNormDetection:
    def test_grad_norm_spike(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            grad_norm_threshold=100.0,
        )
        monitor._record_grad_norm(0, 150.0)
        assert len(monitor._anomalies) == 1
        assert monitor._anomalies[0].anomaly_type == AnomalyType.GRAD_NORM_SPIKE
        assert monitor._anomalies[0].value == 150.0

    def test_normal_grad_norm(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            grad_norm_threshold=100.0,
        )
        monitor._record_grad_norm(0, 50.0)
        assert len(monitor._anomalies) == 0


class TestCaptureStateMachine:
    def test_idle_no_capture(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            ring_buffer_size=0,
        )
        assert monitor._should_capture(0) is False

    def test_capturing_state(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            capture_steps_after=3,
        )
        # Simulate anomaly trigger
        monitor._capture_state = _CaptureState.CAPTURING
        monitor._capture_remaining = 3
        assert monitor._should_capture(0) is True

    def test_ring_buffer_always_captures(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            ring_buffer_size=5,
        )
        assert monitor._ring_buffer_active is True
        assert monitor._should_capture(0) is True

    def test_anomaly_starts_capture_window(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
            capture_steps_after=3,
        )
        assert monitor._capture_state == _CaptureState.IDLE

        monitor._record_loss(0, float("nan"))

        assert monitor._capture_state == _CaptureState.CAPTURING
        assert monitor._capture_remaining == 3

    def test_capture_window_decrements(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            capture_steps_after=2,
        )
        monitor._capture_state = _CaptureState.CAPTURING
        monitor._capture_remaining = 2

        monitor._post_step(10, None)
        assert monitor._capture_remaining == 1
        assert monitor._capture_state == _CaptureState.CAPTURING

        monitor._post_step(11, None)
        assert monitor._capture_remaining == 0
        assert monitor._capture_state == _CaptureState.IDLE

    def test_max_captures_respected(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
            max_captures=2,
        )
        monitor._record_loss(0, float("nan"))
        monitor._record_loss(1, float("nan"))
        assert len(monitor._anomalies) == 2
        assert monitor._capture_count == 2

        # Third anomaly should be ignored
        monitor._record_loss(2, float("nan"))
        assert len(monitor._anomalies) == 2
        assert monitor._capture_count == 2


class TestRingBuffer:
    def test_ring_buffer_eviction(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            ring_buffer_size=2,
        )

        # Create step directories
        dir_0 = tmp_path / "step_0"
        dir_1 = tmp_path / "step_1"
        dir_2 = tmp_path / "step_2"
        dir_0.mkdir()
        dir_1.mkdir()
        dir_2.mkdir()

        monitor._post_step(0, dir_0)
        assert len(monitor._ring_buffer) == 1

        monitor._post_step(1, dir_1)
        assert len(monitor._ring_buffer) == 2

        # Adding step 2 should evict step 0
        monitor._post_step(2, dir_2)
        assert len(monitor._ring_buffer) == 2
        assert not dir_0.exists()
        assert dir_1.exists()
        assert dir_2.exists()


class TestMovingAverage:
    def test_empty_history(self, tmp_path):
        monitor = PrlxTrainingMonitor(output_dir=str(tmp_path))
        assert monitor._moving_average() == 0.0

    def test_average_computation(self, tmp_path):
        monitor = PrlxTrainingMonitor(output_dir=str(tmp_path), window_size=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            monitor._loss_history.append(v)
        assert monitor._moving_average() == 3.0


class TestReport:
    def test_report_generation(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
        )
        monitor._record_loss(0, 1.0)
        monitor._record_loss(1, float("nan"))
        monitor._record_loss(2, 2.0)

        report = monitor.report()
        assert report["total_anomalies"] == 1
        assert report["output_dir"] == str(tmp_path)
        assert report["loss_history_length"] == 3

        # Check that report file was written
        report_path = tmp_path / "training_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            saved = json.load(f)
        assert saved["total_anomalies"] == 1

    def test_anomaly_log_written(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
        )
        monitor._record_loss(0, float("nan"))

        log_path = tmp_path / "anomaly_log.jsonl"
        assert log_path.exists()
        with open(log_path) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["type"] == "nan_inf"
        assert entry["step"] == 0


class TestStepContext:
    def test_step_context_reports_loss(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            nan_detection=True,
        )

        # Mock PrlxTorchWrapper so it doesn't try to import torch
        with mock.patch("prlx.training_monitor.StepContext.__enter__", return_value=None):
            ctx = StepContext(monitor, 0)
            ctx.report_loss(1.5)

        assert monitor._step_losses[0] == 1.5

    def test_step_context_reports_grad_norm(self, tmp_path):
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
        )

        ctx = StepContext(monitor, 5)
        ctx.report_grad_norm(50.0)

        assert monitor._step_grad_norms[5] == 50.0

    def test_step_context_manager_no_capture(self, tmp_path):
        """Step context when not capturing should not start wrapper."""
        monitor = PrlxTrainingMonitor(
            output_dir=str(tmp_path),
            ring_buffer_size=0,
        )

        with monitor.step(0) as ctx:
            ctx.report_loss(1.0)

        assert ctx._wrapper is None

    def test_step_context_does_not_suppress_exceptions(self, tmp_path):
        """Step context should propagate exceptions."""
        monitor = PrlxTrainingMonitor(output_dir=str(tmp_path))

        with pytest.raises(ValueError, match="test error"):
            with monitor.step(0) as ctx:
                raise ValueError("test error")


class TestStepDir:
    def test_step_dir_format(self, tmp_path):
        monitor = PrlxTrainingMonitor(output_dir=str(tmp_path))
        assert monitor._step_dir(42) == tmp_path / "step_42"


class TestExportable:
    def test_importable_from_prlx(self):
        import prlx
        assert hasattr(prlx, "PrlxTrainingMonitor")
        assert prlx.PrlxTrainingMonitor is PrlxTrainingMonitor


class TestCLISubcommands:
    def test_cmd_pytorch_report_json(self, tmp_path, capsys):
        """prlx pytorch report DIR --json should output JSON."""
        from prlx.cli import cmd_pytorch_report

        # Write a fake training_report.json
        report_data = {
            "total_anomalies": 1,
            "anomalies": [{"step": 5, "type": "nan_inf", "value": "NaN",
                          "threshold": 0, "moving_average": 1.0,
                          "timestamp": 1000.0}],
            "captures": 1,
            "output_dir": str(tmp_path),
            "loss_history_length": 10,
        }
        with open(tmp_path / "training_report.json", "w") as f:
            json.dump(report_data, f)

        args = SimpleNamespace(trace_dir=str(tmp_path), json=True)
        result = cmd_pytorch_report(args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_anomalies"] == 1

    def test_cmd_pytorch_report_human(self, tmp_path, capsys):
        """prlx pytorch report DIR should output human-readable summary."""
        from prlx.cli import cmd_pytorch_report

        report_data = {
            "total_anomalies": 2,
            "anomalies": [],
            "captures": 2,
            "output_dir": str(tmp_path),
            "loss_history_length": 50,
        }
        with open(tmp_path / "training_report.json", "w") as f:
            json.dump(report_data, f)

        args = SimpleNamespace(trace_dir=str(tmp_path), json=False)
        result = cmd_pytorch_report(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "PRLX Training Report" in captured.out
        assert "Total anomalies:  2" in captured.out

    def test_cmd_pytorch_report_anomaly_log_fallback(self, tmp_path, capsys):
        """Falls back to anomaly_log.jsonl when training_report.json missing."""
        from prlx.cli import cmd_pytorch_report

        with open(tmp_path / "anomaly_log.jsonl", "w") as f:
            f.write(json.dumps({
                "step": 3, "type": "loss_spike",
                "value": 5.0, "threshold": 4.0,
                "moving_average": 2.0, "timestamp": 1000.0,
            }) + "\n")

        args = SimpleNamespace(trace_dir=str(tmp_path), json=False)
        result = cmd_pytorch_report(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Anomaly Log" in captured.out

    def test_cmd_pytorch_report_missing_dir(self, tmp_path, capsys):
        """Error when trace directory doesn't exist."""
        from prlx.cli import cmd_pytorch_report

        args = SimpleNamespace(
            trace_dir=str(tmp_path / "nonexistent"), json=False)
        result = cmd_pytorch_report(args)
        assert result == 1

    def test_cmd_pytorch_diff_steps_missing_step(self, tmp_path, capsys):
        """Error when step directory doesn't exist."""
        from prlx.cli import cmd_pytorch_diff_steps

        args = SimpleNamespace(
            trace_dir=str(tmp_path), step_a=0, step_b=1,
            map=None, values=False, verbose=False,
        )
        result = cmd_pytorch_diff_steps(args)
        assert result == 1

    def test_cmd_pytorch_watch_missing_script(self, tmp_path, capsys):
        """Error when script doesn't exist."""
        from prlx.cli import cmd_pytorch_watch

        args = SimpleNamespace(
            script=str(tmp_path / "nonexistent.py"),
            output_dir=None, threshold=None, grad_threshold=None,
            window=None, script_args=[],
        )
        result = cmd_pytorch_watch(args)
        assert result == 1
