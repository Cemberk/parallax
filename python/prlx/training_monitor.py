"""
ML Training Loop Integration for PRLX.

Provides PrlxTrainingMonitor that watches training steps for anomalies
(NaN loss, loss spikes, gradient norm explosions) and automatically
captures GPU kernel traces around anomaly windows for differential debugging.
"""

import json
import math
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional

from ._log import get_logger

logger = get_logger(__name__)


class AnomalyType(Enum):
    NAN_INF = "nan_inf"
    LOSS_SPIKE = "loss_spike"
    GRAD_NORM_SPIKE = "grad_norm_spike"


@dataclass
class AnomalyRecord:
    """Record of a detected anomaly during training."""
    step: int
    anomaly_type: AnomalyType
    value: float
    threshold: float
    moving_average: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "type": self.anomaly_type.value,
            "value": self.value,
            "threshold": self.threshold,
            "moving_average": self.moving_average,
            "timestamp": self.timestamp,
        }


class _CaptureState(Enum):
    IDLE = "idle"
    CAPTURING = "capturing"


class StepContext:
    """Context manager for a single training step within PrlxTrainingMonitor.

    Usage::

        with monitor.step(step_num) as ctx:
            loss = train_step(model, data)
            ctx.report_loss(loss.item())
    """

    def __init__(self, monitor: "PrlxTrainingMonitor", step: int):
        self._monitor = monitor
        self._step = step
        self._loss: Optional[float] = None
        self._grad_norm: Optional[float] = None
        self._session_dir: Optional[Path] = None
        self._wrapper = None

    def report_loss(self, loss: float):
        """Record the loss value for this step and check for anomalies."""
        self._loss = loss
        self._monitor._record_loss(self._step, loss)

    def report_grad_norm(self, norm: float):
        """Record the gradient norm for this step and check for anomalies."""
        self._grad_norm = norm
        self._monitor._record_grad_norm(self._step, norm)

    def __enter__(self):
        # Determine if we should capture this step
        should_capture = self._monitor._should_capture(self._step)

        if should_capture:
            self._session_dir = self._monitor._step_dir(self._step)
            self._session_dir.mkdir(parents=True, exist_ok=True)

            # Start trace session via PrlxTorchWrapper if runtime is available
            try:
                from .pytorch_hook import PrlxTorchWrapper
                self._wrapper = PrlxTorchWrapper(
                    name=f"step_{self._step}",
                    session=str(self._session_dir),
                )
                self._wrapper.__enter__()
            except Exception as exc:
                if os.environ.get("PRLX_STRICT_CAPTURE", "0") == "1":
                    raise
                logger.warning(
                    "Failed to initialize step capture for step %d: %s",
                    self._step, exc,
                )
                self._wrapper = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # End trace session
        if self._wrapper is not None:
            self._wrapper.__exit__(exc_type, exc_val, exc_tb)
            self._wrapper = None

        # Check for anomaly after the step completes
        self._monitor._post_step(self._step, self._session_dir)

        # Do not suppress exceptions
        return False


class PrlxTrainingMonitor:
    """Monitor training loops for anomalies and capture GPU kernel traces.

    Usage::

        monitor = prlx.PrlxTrainingMonitor(
            output_dir="./prlx_traces",
            loss_threshold=2.0,
            nan_detection=True,
            grad_norm_threshold=100.0,
        )

        for step in range(num_steps):
            with monitor.step(step) as ctx:
                loss = train_step(model, data)
                ctx.report_loss(loss.item())

        report = monitor.report()

    Args:
        output_dir: Directory to write trace captures.
        loss_threshold: Trigger anomaly if loss > threshold * moving_average.
        nan_detection: Detect NaN/Inf in loss values.
        grad_norm_threshold: Trigger anomaly if gradient norm exceeds this.
        window_size: Number of steps for the moving average window.
        capture_steps_after: Capture N steps after an anomaly is detected.
        ring_buffer_size: Keep traces for the last N steps (0=disabled).
        max_captures: Maximum number of anomaly capture windows.
    """

    def __init__(
        self,
        output_dir: str = "./prlx_traces",
        loss_threshold: float = 2.0,
        nan_detection: bool = True,
        grad_norm_threshold: float = 100.0,
        window_size: int = 50,
        capture_steps_after: int = 3,
        ring_buffer_size: int = 0,
        max_captures: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.loss_threshold = loss_threshold
        self.nan_detection = nan_detection
        self.grad_norm_threshold = grad_norm_threshold
        self.window_size = window_size
        self.capture_steps_after = capture_steps_after
        self.ring_buffer_size = ring_buffer_size
        self.max_captures = max_captures

        # Internal state
        self._loss_history: Deque[float] = deque(maxlen=window_size)
        self._grad_norm_history: Deque[float] = deque(maxlen=window_size)
        self._anomalies: List[AnomalyRecord] = []
        self._capture_state = _CaptureState.IDLE
        self._capture_remaining = 0
        self._capture_count = 0
        self._step_losses: Dict[int, float] = {}
        self._step_grad_norms: Dict[int, float] = {}

        # Ring buffer for "before anomaly" traces
        self._ring_buffer: Deque[Path] = deque(maxlen=ring_buffer_size if ring_buffer_size > 0 else 1)
        self._ring_buffer_active = ring_buffer_size > 0

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "PrlxTrainingMonitor":
        """Create a monitor from environment variables (for use with CLI).

        Reads PRLX_PYTORCH_WATCH env var and related settings.
        """
        output_dir = os.environ.get("PRLX_TRACE_DIR", "./prlx_traces")
        threshold = float(os.environ.get("PRLX_LOSS_THRESHOLD", "2.0"))
        grad_threshold = float(os.environ.get("PRLX_GRAD_NORM_THRESHOLD", "100.0"))
        window = int(os.environ.get("PRLX_WINDOW_SIZE", "50"))
        capture_after = int(os.environ.get("PRLX_CAPTURE_STEPS_AFTER", "3"))

        return cls(
            output_dir=output_dir,
            loss_threshold=threshold,
            grad_norm_threshold=grad_threshold,
            window_size=window,
            capture_steps_after=capture_after,
        )

    def step(self, step_num: int) -> StepContext:
        """Create a context manager for a training step.

        Args:
            step_num: The step/iteration number.

        Returns:
            StepContext that should be used as a context manager.
        """
        return StepContext(self, step_num)

    def report(self) -> dict:
        """Generate a summary report of the training monitoring session.

        Returns:
            Dictionary with anomalies, loss history, and capture info.
        """
        report = {
            "total_anomalies": len(self._anomalies),
            "anomalies": [a.to_dict() for a in self._anomalies],
            "captures": self._capture_count,
            "output_dir": str(self.output_dir),
            "loss_history_length": len(self._step_losses),
        }

        # Write report to file
        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    # ---- Internal methods ----

    def _step_dir(self, step: int) -> Path:
        return self.output_dir / f"step_{step}"

    def _should_capture(self, step: int) -> bool:
        """Determine if this step should have trace capture active."""
        # Always capture if in post-anomaly window
        if self._capture_state == _CaptureState.CAPTURING:
            return True

        # Ring buffer mode: always capture
        if self._ring_buffer_active:
            return True

        return False

    def _record_loss(self, step: int, loss: float):
        """Record a loss value and check for anomalies."""
        self._step_losses[step] = loss

        # NaN/Inf detection
        if self.nan_detection and (math.isnan(loss) or math.isinf(loss)):
            ma = self._moving_average()
            self._trigger_anomaly(AnomalyRecord(
                step=step,
                anomaly_type=AnomalyType.NAN_INF,
                value=loss,
                threshold=0.0,
                moving_average=ma,
            ))
            return

        # Loss spike detection (only after warmup)
        if len(self._loss_history) >= self.window_size:
            ma = self._moving_average()
            if ma > 0 and loss > self.loss_threshold * ma:
                self._trigger_anomaly(AnomalyRecord(
                    step=step,
                    anomaly_type=AnomalyType.LOSS_SPIKE,
                    value=loss,
                    threshold=self.loss_threshold * ma,
                    moving_average=ma,
                ))

        self._loss_history.append(loss)

    def _record_grad_norm(self, step: int, norm: float):
        """Record a gradient norm and check for anomalies."""
        self._step_grad_norms[step] = norm

        if norm > self.grad_norm_threshold:
            ma = self._moving_average()
            self._trigger_anomaly(AnomalyRecord(
                step=step,
                anomaly_type=AnomalyType.GRAD_NORM_SPIKE,
                value=norm,
                threshold=self.grad_norm_threshold,
                moving_average=ma,
            ))

        self._grad_norm_history.append(norm)

    def _moving_average(self) -> float:
        """Compute moving average of recent loss values."""
        if not self._loss_history:
            return 0.0
        return sum(self._loss_history) / len(self._loss_history)

    def _trigger_anomaly(self, record: AnomalyRecord):
        """Handle a detected anomaly."""
        if self._capture_count >= self.max_captures:
            return

        self._anomalies.append(record)

        # Transition to capturing state
        self._capture_state = _CaptureState.CAPTURING
        self._capture_remaining = self.capture_steps_after
        self._capture_count += 1

        # Write anomaly log
        log_path = self.output_dir / "anomaly_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _post_step(self, step: int, session_dir: Optional[Path]):
        """Called after each step completes."""
        # Manage ring buffer
        if self._ring_buffer_active and session_dir and session_dir.exists():
            # Evict oldest entry if buffer is full
            if len(self._ring_buffer) >= self._ring_buffer.maxlen:
                oldest = self._ring_buffer[0]
                if oldest.exists():
                    shutil.rmtree(oldest, ignore_errors=True)
            self._ring_buffer.append(session_dir)

        # Manage post-anomaly capture window
        if self._capture_state == _CaptureState.CAPTURING:
            self._capture_remaining -= 1
            if self._capture_remaining <= 0:
                self._capture_state = _CaptureState.IDLE
