"""
Library and binary discovery for gpu-diffdbg components.

Searches for:
    - libGpuDiffDbgPass.so       (LLVM pass plugin)
    - libgddbg_runtime.so        (shared runtime library)
    - gddbg_runtime_nvptx.bc     (NVPTX bitcode for Triton linking)
    - gddbg-diff                 (Rust differ binary)
    - opt-18 / llvm-link-18      (LLVM tools for pass pipeline)
"""

import os
import shutil
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    """Find the gpu-diffdbg project root (where CMakeLists.txt lives)."""
    # Walk up from this file's location
    current = Path(__file__).resolve().parent  # python/gddbg/
    for _ in range(5):
        current = current.parent
        if (current / "CMakeLists.txt").exists() and (current / "differ").exists():
            return current
    return Path(__file__).resolve().parent.parent.parent


def find_pass_plugin() -> Optional[Path]:
    """Find the libGpuDiffDbgPass.so LLVM pass plugin."""
    candidates = []

    # Check GPU_DIFFDBG_HOME env var
    home = os.environ.get("GPU_DIFFDBG_HOME")
    if home:
        candidates.append(Path(home) / "lib" / "libGpuDiffDbgPass.so")
        candidates.append(Path(home) / "build" / "lib" / "pass" / "libGpuDiffDbgPass.so")

    # Check relative to project
    root = _project_root()
    candidates.extend([
        root / "build" / "lib" / "pass" / "libGpuDiffDbgPass.so",
        root / "lib" / "pass" / "libGpuDiffDbgPass.so",
    ])

    for c in candidates:
        if c.exists():
            return c.resolve()

    return None


def find_runtime_library() -> Optional[Path]:
    """Find the libgddbg_runtime shared library."""
    candidates = []

    home = os.environ.get("GPU_DIFFDBG_HOME")
    if home:
        candidates.append(Path(home) / "lib" / "libgddbg_runtime_shared.so")
        candidates.append(Path(home) / "build" / "lib" / "runtime" / "libgddbg_runtime_shared.so")

    root = _project_root()
    candidates.extend([
        root / "build" / "lib" / "runtime" / "libgddbg_runtime_shared.so",
    ])

    for c in candidates:
        if c.exists():
            return c.resolve()

    return None


def find_runtime_bitcode() -> Optional[Path]:
    """Find the NVPTX bitcode for Triton linking."""
    candidates = []

    home = os.environ.get("GPU_DIFFDBG_HOME")
    if home:
        candidates.append(Path(home) / "lib" / "gddbg_runtime_nvptx.bc")
        candidates.append(
            Path(home) / "build" / "lib" / "runtime" / "gddbg_runtime_nvptx.bc"
        )

    root = _project_root()
    candidates.extend([
        root / "build" / "lib" / "runtime" / "gddbg_runtime_nvptx.bc",
    ])

    for c in candidates:
        if c.exists():
            return c.resolve()

    return None


def find_opt_binary() -> Optional[Path]:
    """Find opt (LLVM optimizer) for running the instrumentation pass.
    Prefers LLVM 20 (closest to Triton's bundled LLVM), falls back to 18."""
    for name in ("opt-20", "opt-18", "opt"):
        path = shutil.which(name)
        if path:
            return Path(path)
    return None


def find_llvm_link_binary() -> Optional[Path]:
    """Find llvm-link for linking NVPTX bitcode modules.
    Prefers LLVM 20, falls back to 18."""
    for name in ("llvm-link-20", "llvm-link-18", "llvm-link"):
        path = shutil.which(name)
        if path:
            return Path(path)
    return None


def find_differ_binary() -> Optional[Path]:
    """Find the gddbg-diff Rust binary."""
    import subprocess

    root = _project_root()
    candidates = [
        root / "differ" / "target" / "release" / "gddbg-diff",
        root / "differ" / "target" / "debug" / "gddbg-diff",
    ]

    home = os.environ.get("GPU_DIFFDBG_HOME")
    if home:
        candidates.insert(0, Path(home) / "bin" / "gddbg-diff")

    for c in candidates:
        if c.exists():
            return c.resolve()

    # Try PATH
    try:
        result = subprocess.run(
            ["which", "gddbg-diff"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except FileNotFoundError:
        pass

    return None
