"""
Library and binary discovery for prlx components.

Search priority for each component:
    1. PRLX_HOME env var (explicit override)
    2. Bundled package data (inside installed wheel)
    3. Project root fallback (development / editable install)
    4. System PATH (for LLVM tools and prlx-diff)
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

# Package data directory (populated inside wheels, empty in dev mode)
_PACKAGE_DATA = Path(__file__).resolve().parent / "data"


def _project_root() -> Optional[Path]:
    """Find the prlx project root (where CMakeLists.txt + differ/ live).

    Returns None if not in a source tree (e.g., installed via pip).
    """
    current = Path(__file__).resolve().parent  # python/prlx/
    for _ in range(5):
        current = current.parent
        if (current / "CMakeLists.txt").exists() and (current / "differ").exists():
            return current
    return None


def _detect_llvm_version() -> Optional[int]:
    """Detect the major LLVM version available on the system.

    Checks opt-20, opt-19, opt-18, opt in order and parses --version output.
    """
    for name in ("opt-20", "opt-19", "opt-18", "opt"):
        path = shutil.which(name)
        if path:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.splitlines():
                    m = re.search(r"(\d+)\.\d+", line)
                    if m:
                        return int(m.group(1))
            except (subprocess.SubprocessError, OSError):
                continue
    return None


def find_pass_plugin() -> Optional[Path]:
    """Find libPrlxPass.so matching the system's LLVM version.

    For installed wheels, versioned variants (libPrlxPass.llvm20.so) are
    selected based on the detected LLVM version.
    """
    # 1. Explicit override
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libPrlxPass.so",
            Path(home) / "build" / "lib" / "pass" / "libPrlxPass.so",
        ]:
            if p.exists():
                return p.resolve()

    # 2. Bundled package data (versioned by LLVM)
    llvm_ver = _detect_llvm_version()
    if llvm_ver:
        p = _PACKAGE_DATA / "lib" / f"libPrlxPass.llvm{llvm_ver}.so"
        if p.exists():
            return p.resolve()

    # Bundled unversioned fallback
    p = _PACKAGE_DATA / "lib" / "libPrlxPass.so"
    if p.exists():
        return p.resolve()

    # 3. Project root (dev mode)
    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "pass" / "libPrlxPass.so"
            if p.exists():
                return p.resolve()

    return None


def find_runtime_library() -> Optional[Path]:
    """Find the libprlx_runtime_shared.so shared library."""
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libprlx_runtime_shared.so",
            Path(home) / "build" / "lib" / "runtime" / "libprlx_runtime_shared.so",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "libprlx_runtime_shared.so"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "runtime" / "libprlx_runtime_shared.so"
            if p.exists():
                return p.resolve()

    return None


def find_static_runtime() -> Optional[Path]:
    """Find the libprlx_runtime.a static archive."""
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "libprlx_runtime.a",
            Path(home) / "build" / "lib" / "runtime" / "libprlx_runtime.a",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "libprlx_runtime.a"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "runtime" / "libprlx_runtime.a"
            if p.exists():
                return p.resolve()

    return None


def find_runtime_bitcode() -> Optional[Path]:
    """Find the NVPTX bitcode for Triton linking."""
    home = os.environ.get("PRLX_HOME")
    if home:
        for p in [
            Path(home) / "lib" / "prlx_runtime_nvptx.bc",
            Path(home) / "build" / "lib" / "runtime" / "prlx_runtime_nvptx.bc",
        ]:
            if p.exists():
                return p.resolve()

    p = _PACKAGE_DATA / "lib" / "prlx_runtime_nvptx.bc"
    if p.exists():
        return p.resolve()

    root = _project_root()
    if root:
        for build_dir in ("build-prlx", "build"):
            p = root / build_dir / "lib" / "runtime" / "prlx_runtime_nvptx.bc"
            if p.exists():
                return p.resolve()

    return None


def find_include_dirs() -> List[Path]:
    """Find directories containing prlx headers (prlx_runtime.h, trace_format.h).

    Returns a list of include paths to pass to the compiler via -I.
    """
    # Bundled
    inc = _PACKAGE_DATA / "include"
    if inc.exists() and (inc / "trace_format.h").exists():
        return [inc]

    # Project root (dev mode) — headers are in two dirs
    root = _project_root()
    if root:
        dirs = []
        runtime_dir = root / "lib" / "runtime"
        common_dir = root / "lib" / "common"
        if runtime_dir.exists():
            dirs.append(runtime_dir)
        if common_dir.exists():
            dirs.append(common_dir)
        if dirs:
            return dirs

    return []


def find_opt_binary() -> Optional[Path]:
    """Find opt (LLVM optimizer) for running the instrumentation pass."""
    for name in ("opt-20", "opt-19", "opt-18", "opt"):
        path = shutil.which(name)
        if path:
            return Path(path)
    return None


def find_llvm_link_binary() -> Optional[Path]:
    """Find llvm-link for linking NVPTX bitcode modules."""
    for name in ("llvm-link-20", "llvm-link-19", "llvm-link-18", "llvm-link"):
        path = shutil.which(name)
        if path:
            return Path(path)
    return None


def find_differ_binary() -> Optional[Path]:
    """Find the prlx-diff Rust binary."""
    # 1. Explicit override
    home = os.environ.get("PRLX_HOME")
    if home:
        p = Path(home) / "bin" / "prlx-diff"
        if p.exists():
            return p.resolve()

    # 2. Bundled package data
    p = _PACKAGE_DATA / "bin" / "prlx-diff"
    if p.exists():
        return p.resolve()

    # 3. Project root (dev mode)
    root = _project_root()
    if root:
        for variant in ("release", "debug"):
            p = root / "differ" / "target" / variant / "prlx-diff"
            if p.exists():
                return p.resolve()

    # 4. System PATH
    path = shutil.which("prlx-diff")
    if path:
        return Path(path)

    return None
