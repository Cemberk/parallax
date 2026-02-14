"""
Triton compiler integration for gpu-diffdbg.

Hooks into Triton's compilation pipeline to inject the gpu-diffdbg
LLVM instrumentation pass. Uses environment variables and launch
wrappers for maximum compatibility across Triton versions.

Usage:
    import gddbg
    gddbg.integrate_with_triton()

    @triton.jit
    def my_kernel(...):
        ...

    # Kernels are now automatically instrumented
"""

import os
import sys
import functools
from pathlib import Path
from typing import Optional

from ._find_lib import find_pass_plugin, find_runtime_library


_installed = False
_original_launch = None


def install(pass_plugin: Optional[str] = None):
    """
    Install gpu-diffdbg instrumentation into Triton's compilation pipeline.

    This works by:
    1. Setting LLVM_PASS_PLUGINS env var so Triton's LLVM backend loads our pass
    2. Wrapping kernel launches with pre/post hooks for trace buffer management

    Args:
        pass_plugin: Explicit path to libGpuDiffDbgPass.so. Auto-detected if None.
    """
    global _installed
    if _installed:
        return

    # Find the pass plugin
    if pass_plugin is None:
        plugin = find_pass_plugin()
    else:
        plugin = Path(pass_plugin)

    if plugin is None or not plugin.exists():
        raise FileNotFoundError(
            "libGpuDiffDbgPass.so not found. Build with:\n"
            "  cmake --build build\n"
            "Or set GPU_DIFFDBG_HOME to the install directory."
        )

    # Set LLVM pass plugin environment variable
    # Triton uses LLVM internally and respects LLVM_PASS_PLUGINS
    existing = os.environ.get("LLVM_PASS_PLUGINS", "")
    if str(plugin) not in existing:
        os.environ["LLVM_PASS_PLUGINS"] = (
            f"{existing};{plugin}" if existing else str(plugin)
        )

    print(f"[gddbg] LLVM pass plugin: {plugin}", file=sys.stderr)

    # Try to wrap Triton's kernel launch
    _wrap_triton_launch()

    _installed = True
    print("[gddbg] Triton integration installed", file=sys.stderr)


def _wrap_triton_launch():
    """
    Wrap Triton's JIT kernel launch to insert pre/post trace hooks.

    This is a best-effort monkey-patch. If Triton's internals change,
    the instrumentation pass still works (via LLVM_PASS_PLUGINS), but
    trace buffer management won't be automatic.
    """
    global _original_launch

    try:
        import triton
        from triton.runtime import JITFunction
    except ImportError:
        print(
            "[gddbg] Triton not installed. LLVM_PASS_PLUGINS set for manual use.",
            file=sys.stderr,
        )
        return

    # Check if already wrapped
    if _original_launch is not None:
        return

    # Try to load the runtime library for pre/post launch hooks
    rt_lib = find_runtime_library()
    if rt_lib is None:
        print(
            "[gddbg] Runtime library not found. Trace buffers must be managed manually.",
            file=sys.stderr,
        )
        return

    try:
        from .runtime import GddbgRuntime
        runtime = GddbgRuntime(str(rt_lib))
    except Exception as e:
        print(f"[gddbg] Could not load runtime: {e}", file=sys.stderr)
        return

    # Wrap JITFunction.run to inject pre/post hooks
    _original_launch = JITFunction.run

    @functools.wraps(_original_launch)
    def instrumented_run(self, *args, **kwargs):
        grid = kwargs.get("grid", (1, 1, 1))
        # Normalize grid to 3-tuple
        if callable(grid):
            # Triton often passes grid as a lambda; we can't resolve it here
            # without the meta dict. Fall back to a placeholder.
            grid_dim = (1, 1, 1)
        elif isinstance(grid, (tuple, list)):
            grid_dim = tuple(grid) + (1,) * (3 - len(grid))
        else:
            grid_dim = (int(grid), 1, 1)

        # Get block size from kernel metadata if available
        block_dim = (kwargs.get("num_warps", 4) * 32, 1, 1)

        kernel_name = getattr(self, "fn.__name__", "triton_kernel")

        # Enable tracing for this launch
        if os.environ.get("GDDBG_ENABLED", "1") != "0":
            try:
                runtime.pre_launch(kernel_name, grid_dim, block_dim)
            except Exception as e:
                print(f"[gddbg] pre_launch failed: {e}", file=sys.stderr)

        result = _original_launch(self, *args, **kwargs)

        if os.environ.get("GDDBG_ENABLED", "1") != "0":
            try:
                runtime.post_launch()
            except Exception as e:
                print(f"[gddbg] post_launch failed: {e}", file=sys.stderr)

        return result

    JITFunction.run = instrumented_run
    print("[gddbg] Triton kernel launch wrapper installed", file=sys.stderr)


def uninstall():
    """Remove the Triton integration hook."""
    global _installed, _original_launch

    if _original_launch is not None:
        try:
            from triton.runtime import JITFunction
            JITFunction.run = _original_launch
        except ImportError:
            pass
        _original_launch = None

    # Clean up env var
    plugins = os.environ.get("LLVM_PASS_PLUGINS", "")
    plugin = find_pass_plugin()
    if plugin and str(plugin) in plugins:
        parts = [p for p in plugins.split(";") if p and str(plugin) not in p]
        if parts:
            os.environ["LLVM_PASS_PLUGINS"] = ";".join(parts)
        else:
            os.environ.pop("LLVM_PASS_PLUGINS", None)

    _installed = False
