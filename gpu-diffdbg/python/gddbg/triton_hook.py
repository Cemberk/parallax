"""
Triton compiler integration for gpu-diffdbg.

Hooks into Triton's compilation pipeline to inject the gpu-diffdbg
LLVM instrumentation pass. Works by intercepting the LLVM IR between
Triton's make_llir() and make_ptx() stages, linking the device-side
runtime bitcode, and running the instrumentation pass via opt.

Usage:
    import gddbg
    gddbg.enable()

    @triton.jit
    def my_kernel(...):
        ...

    # Kernels are now automatically instrumented
"""

import os
import sys
import subprocess
import tempfile
import functools
from pathlib import Path
from typing import Optional

from ._find_lib import (
    find_pass_plugin,
    find_runtime_bitcode,
    find_runtime_library,
    find_opt_binary,
    find_llvm_link_binary,
)


_installed = False
_original_launch = None
_pass_plugin: Optional[Path] = None
_runtime_bc: Optional[Path] = None
_opt_bin: Optional[Path] = None
_llvm_link_bin: Optional[Path] = None


def install(pass_plugin: Optional[str] = None, verbose: bool = True):
    """
    Install gpu-diffdbg instrumentation into Triton's compilation pipeline.

    This works by:
    1. Hooking Triton's pipeline stages to intercept LLVM IR after make_llir()
    2. Linking the gpu-diffdbg device runtime bitcode into the module
    3. Running the gpu-diffdbg LLVM instrumentation pass via opt
    4. Wrapping kernel launches with pre/post hooks for trace buffer management

    Args:
        pass_plugin: Explicit path to libGpuDiffDbgPass.so. Auto-detected if None.
        verbose: Print status messages to stderr.
    """
    global _installed, _pass_plugin, _runtime_bc, _opt_bin, _llvm_link_bin
    if _installed:
        return

    # Find all required components
    _pass_plugin = Path(pass_plugin) if pass_plugin else find_pass_plugin()
    _runtime_bc = find_runtime_bitcode()
    _opt_bin = find_opt_binary()
    _llvm_link_bin = find_llvm_link_binary()

    # Validate
    missing = []
    if not _pass_plugin or not _pass_plugin.exists():
        missing.append(
            "libGpuDiffDbgPass.so (build with: cmake --build build)"
        )
    if not _runtime_bc or not _runtime_bc.exists():
        missing.append(
            "gddbg_runtime_nvptx.bc (build with: cmake --build build)"
        )
    if not _opt_bin:
        missing.append("opt (install with: apt install llvm-20)")
    if not _llvm_link_bin:
        missing.append("llvm-link (install with: apt install llvm-20)")

    if missing:
        raise FileNotFoundError(
            "Missing components for Triton integration:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    if verbose:
        print(f"[gddbg] Pass plugin: {_pass_plugin}", file=sys.stderr)
        print(f"[gddbg] Runtime BC:  {_runtime_bc}", file=sys.stderr)
        print(f"[gddbg] opt:         {_opt_bin}", file=sys.stderr)
        print(f"[gddbg] llvm-link:   {_llvm_link_bin}", file=sys.stderr)

    # Hook into Triton's compilation pipeline
    _hook_triton_stages(verbose)

    # Wrap kernel launches with trace buffer management
    _wrap_triton_launch(verbose)

    _installed = True
    if verbose:
        print("[gddbg] Triton integration installed", file=sys.stderr)


def _instrument_llvm_ir(llvm_ir: str) -> str:
    """
    Instrument LLVM IR by linking runtime bitcode and running the pass.

    Pipeline: input.ll → llvm-link with runtime BC → opt with pass → output.ll

    The runtime bitcode is compiled with -fcuda-flush-denormals-to-zero so its
    nvvm-reflect-ftz flag (=1) matches Triton's. Data layout differences between
    Triton's extended layout and clang's default are harmless warnings suppressed
    by --suppress-warnings.
    """
    with tempfile.TemporaryDirectory(prefix="gddbg_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.ll")
        linked_path = os.path.join(tmpdir, "linked.ll")
        output_path = os.path.join(tmpdir, "output.ll")

        with open(input_path, "w") as f:
            f.write(llvm_ir)

        # Step 1: Link kernel IR with device runtime bitcode
        link_cmd = [
            str(_llvm_link_bin),
            input_path,
            str(_runtime_bc),
            "-S",
            "-o", linked_path,
            "--suppress-warnings",
        ]
        result = subprocess.run(
            link_cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(
                f"[gddbg] WARNING: llvm-link failed, skipping instrumentation:\n"
                f"  {result.stderr[:500]}",
                file=sys.stderr,
            )
            return llvm_ir

        # Step 2: Run instrumentation pass via opt
        opt_cmd = [
            str(_opt_bin),
            "-load-pass-plugin", str(_pass_plugin),
            "-passes=gpu-diffdbg",
            linked_path,
            "-S",
            "-o", output_path,
        ]
        result = subprocess.run(
            opt_cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(
                f"[gddbg] WARNING: opt pass failed, skipping instrumentation:\n"
                f"  {result.stderr[:500]}",
                file=sys.stderr,
            )
            return llvm_ir

        with open(output_path) as f:
            instrumented = f.read()

        # Report instrumentation results from opt stderr
        for line in result.stderr.splitlines():
            if "[gddbg]" in line:
                print(line.strip(), file=sys.stderr)

        return instrumented


def _hook_triton_stages(verbose: bool):
    """
    Hook into Triton's compilation stages to intercept LLVM IR.

    Uses Triton 3.x's add_stages_inspection_hook API to wrap the 'llir'
    stage. Falls back to monkey-patching CUDABackend.make_llir if the
    hook API is not available.
    """
    hooked = False

    # Try the official stages hook API first (Triton >= 3.x)
    try:
        from triton import knobs
        if hasattr(knobs, "runtime") and hasattr(
            knobs.runtime, "add_stages_inspection_hook"
        ):
            _hook_via_stages_api(verbose)
            hooked = True
    except ImportError:
        pass

    if not hooked:
        # Try monkey-patching CUDABackend.make_llir
        _hook_via_monkey_patch(verbose)


def _hook_via_stages_api(verbose: bool):
    """Hook using Triton's official stages inspection API."""
    from triton import knobs

    def gddbg_stages_hook(backend, stages, options, language, capability):
        if "llir" not in stages:
            return

        original_llir = stages["llir"]

        def instrumented_llir(src, metadata):
            llvm_ir = original_llir(src, metadata)

            if os.environ.get("GDDBG_ENABLED", "1") == "0":
                return llvm_ir

            return _instrument_llvm_ir(llvm_ir)

        stages["llir"] = instrumented_llir

    knobs.runtime.add_stages_inspection_hook = gddbg_stages_hook

    if verbose:
        print(
            "[gddbg] Hooked Triton stages API (llir interception)",
            file=sys.stderr,
        )


def _hook_via_monkey_patch(verbose: bool):
    """Fallback: monkey-patch CUDABackend.make_llir directly."""
    try:
        from triton.backends.nvidia.compiler import CUDABackend
    except ImportError:
        if verbose:
            print(
                "[gddbg] Cannot find CUDABackend. "
                "Setting LLVM_PASS_PLUGINS as fallback.",
                file=sys.stderr,
            )
        _set_llvm_env_var()
        return

    original_make_llir = CUDABackend.make_llir

    def patched_make_llir(self, src, metadata, options, capability):
        llvm_ir = original_make_llir(self, src, metadata, options, capability)

        if os.environ.get("GDDBG_ENABLED", "1") == "0":
            return llvm_ir

        return _instrument_llvm_ir(llvm_ir)

    CUDABackend.make_llir = patched_make_llir

    if verbose:
        print(
            "[gddbg] Monkey-patched CUDABackend.make_llir",
            file=sys.stderr,
        )


def _set_llvm_env_var():
    """Last resort: set LLVM_PASS_PLUGINS env var."""
    if _pass_plugin:
        existing = os.environ.get("LLVM_PASS_PLUGINS", "")
        if str(_pass_plugin) not in existing:
            os.environ["LLVM_PASS_PLUGINS"] = (
                f"{existing};{_pass_plugin}" if existing else str(_pass_plugin)
            )
        print(
            "[gddbg] Set LLVM_PASS_PLUGINS (fallback, may not work with "
            "Triton's bundled LLVM)",
            file=sys.stderr,
        )


def _wrap_triton_launch(verbose: bool):
    """
    Wrap Triton's JIT kernel launch to insert pre/post trace hooks.

    Manages trace buffer allocation/deallocation around each kernel launch.
    Uses the shared runtime library (libgddbg_runtime_shared.so) via ctypes.
    """
    global _original_launch

    try:
        from triton.runtime import JITFunction
    except ImportError:
        if verbose:
            print(
                "[gddbg] Triton not installed. Launch wrapper skipped.",
                file=sys.stderr,
            )
        return

    if _original_launch is not None:
        return

    # Try to load the runtime library
    rt_lib = find_runtime_library()
    if rt_lib is None:
        if verbose:
            print(
                "[gddbg] Runtime library not found. "
                "Trace buffers must be managed manually.",
                file=sys.stderr,
            )
        return

    try:
        from .runtime import GddbgRuntime
        runtime = GddbgRuntime(str(rt_lib))
    except Exception as e:
        if verbose:
            print(f"[gddbg] Could not load runtime: {e}", file=sys.stderr)
        return

    _original_launch = JITFunction.run

    @functools.wraps(_original_launch)
    def instrumented_run(self, *args, **kwargs):
        if os.environ.get("GDDBG_ENABLED", "1") == "0":
            return _original_launch(self, *args, **kwargs)

        grid = kwargs.get("grid", (1, 1, 1))
        if callable(grid):
            grid_dim = (1, 1, 1)
        elif isinstance(grid, (tuple, list)):
            grid_dim = tuple(grid) + (1,) * (3 - len(grid))
        else:
            grid_dim = (int(grid), 1, 1)

        block_dim = (kwargs.get("num_warps", 4) * 32, 1, 1)
        kernel_name = getattr(self, "__name__", "triton_kernel")

        try:
            runtime.pre_launch(kernel_name, grid_dim, block_dim)
        except Exception as e:
            print(f"[gddbg] pre_launch failed: {e}", file=sys.stderr)

        result = _original_launch(self, *args, **kwargs)

        try:
            runtime.post_launch()
        except Exception as e:
            print(f"[gddbg] post_launch failed: {e}", file=sys.stderr)

        return result

    JITFunction.run = instrumented_run
    if verbose:
        print("[gddbg] Triton kernel launch wrapper installed", file=sys.stderr)


def uninstall():
    """Remove all Triton integration hooks."""
    global _installed, _original_launch

    # Restore kernel launch
    if _original_launch is not None:
        try:
            from triton.runtime import JITFunction
            JITFunction.run = _original_launch
        except ImportError:
            pass
        _original_launch = None

    # Remove stages hook
    try:
        from triton import knobs
        if hasattr(knobs, "runtime"):
            knobs.runtime.add_stages_inspection_hook = None
    except ImportError:
        pass

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
