"""
PRLX - Differential debugger for CUDA/Triton GPU kernels.

Public API:
    enable()                 - Hook into Triton's compilation pipeline
    integrate_with_triton()  - Alias for enable()
    read_trace(path)         - Read a .prlx trace file
    diff_traces(a, b)        - Compare two trace files using the Rust differ
    session(name)            - Context manager for multi-kernel pipeline tracing
"""

__version__ = "0.1.0"

from .trace_reader import read_trace, TraceData
from ._find_lib import find_differ_binary


def enable(**kwargs):
    """
    Enable prlx instrumentation for Triton kernels.

    This intercepts Triton's compilation pipeline to inject the prlx
    LLVM instrumentation pass, and wraps kernel launches with trace buffer
    management hooks.

    Usage:
        import prlx
        prlx.enable()

        @triton.jit
        def my_kernel(...):
            ...
        # All subsequent Triton kernels are automatically instrumented.

    Args:
        pass_plugin: Explicit path to libPrlxPass.so.
        verbose: Print status messages (default True).
    """
    from .triton_hook import install
    install(**kwargs)


def integrate_with_triton(**kwargs):
    """Alias for enable(). Hook into Triton's compilation pipeline."""
    enable(**kwargs)


def diff_traces(trace_a: str, trace_b: str, **kwargs) -> int:
    """
    Run the Rust differ on two trace files.

    Args:
        trace_a: Path to baseline trace file.
        trace_b: Path to comparison trace file.
        **kwargs: Additional options passed to prlx-diff CLI
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
            "prlx-diff binary not found. "
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


def session(name: str, **kwargs):
    """
    Context manager for session-based multi-kernel pipeline tracing.

    Usage:
        with prlx.session("my_pipeline"):
            # launch multiple kernels
            pass
        # session.json manifest is written to my_pipeline/

    Args:
        name: Session directory name.
        **kwargs: Passed to PrlxRuntime constructor.
    """
    from .runtime import session as _session
    return _session(name, **kwargs)
