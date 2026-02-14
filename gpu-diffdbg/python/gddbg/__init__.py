"""
GPU DiffDbg - Differential debugger for CUDA/Triton GPU kernels.

Public API:
    integrate_with_triton()  - Hook into Triton's compilation pipeline
    read_trace(path)         - Read a .gddbg trace file
    diff_traces(a, b)        - Compare two trace files using the Rust differ
"""

__version__ = "0.1.0"

from .trace_reader import read_trace, TraceData
from ._find_lib import find_differ_binary


def integrate_with_triton():
    """Install the gpu-diffdbg instrumentation hook into Triton's compiler."""
    from .triton_hook import install
    install()


def diff_traces(trace_a: str, trace_b: str, **kwargs) -> int:
    """
    Run the Rust differ on two trace files.

    Args:
        trace_a: Path to baseline trace file.
        trace_b: Path to comparison trace file.
        **kwargs: Additional options passed to gddbg-diff CLI
            map: Path to site map JSON
            values: bool, compare operand values
            tui: bool, launch interactive TUI
            history: bool, show value history

    Returns:
        Exit code from the differ (0 = identical, 1 = divergences found).
    """
    import subprocess

    differ = find_differ_binary()
    if differ is None:
        raise FileNotFoundError(
            "gddbg-diff binary not found. "
            "Build with: cd differ && cargo build --release"
        )

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
