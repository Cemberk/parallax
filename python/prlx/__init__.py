"""PRLX - Differential debugger for CUDA/Triton GPU kernels."""

try:
    from importlib.metadata import version as _meta_version
    __version__ = _meta_version("prlx")
except Exception:
    __version__ = "0.0.0.dev0"

from .trace_reader import read_trace, TraceData
from ._find_lib import find_differ_binary


def enable(**kwargs):
    """Hook into Triton's compilation pipeline to instrument kernels."""
    from .triton_hook import install
    install(**kwargs)


def integrate_with_triton(**kwargs):
    """Alias for enable()."""
    enable(**kwargs)


def diff_traces(trace_a: str, trace_b: str, **kwargs) -> int:
    """Run the Rust differ on two trace files. Returns exit code (0=identical, 1=divergent)."""
    import subprocess

    differ = find_differ_binary()
    if differ is None:
        raise FileNotFoundError("prlx-diff binary not found. Install with: pip install prlx")

    cmd = [str(differ), str(trace_a), str(trace_b)]

    if kwargs.get("map"):
        cmd.extend(["--map", str(kwargs["map"])])
    if kwargs.get("values"):
        cmd.append("--values")
    if kwargs.get("tui"):
        cmd.append("--tui")
    if kwargs.get("history"):
        cmd.append("--history")

    return subprocess.call(cmd)


def session(name: str, **kwargs):
    """Context manager for multi-kernel pipeline tracing."""
    from .runtime import session as _session
    return _session(name, **kwargs)
