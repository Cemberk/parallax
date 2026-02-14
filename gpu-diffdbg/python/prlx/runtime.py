"""
ctypes FFI wrapper around libprlx_runtime_shared.so.

Provides Python access to the prlx runtime for:
    - Initializing trace buffers
    - Pre/post kernel launch hooks
    - Context manager for automatic tracing

Requires the shared library to be built:
    cmake --build build --target prlx_runtime_shared
"""

import ctypes
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

from ._find_lib import find_runtime_library


class PrlxRuntime:
    """
    Python wrapper around the prlx C runtime.

    Usage:
        rt = PrlxRuntime()
        rt.init()
        rt.pre_launch("my_kernel", (1, 1, 1), (256, 1, 1))
        # ... launch CUDA kernel ...
        rt.post_launch()
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Load the shared runtime library.

        Args:
            lib_path: Explicit path to libprlx_runtime_shared.so.
                      If None, searches standard locations.
        """
        if lib_path is None:
            found = find_runtime_library()
            if found is None:
                raise FileNotFoundError(
                    "libprlx_runtime_shared.so not found. "
                    "Build with: cmake --build build --target prlx_runtime_shared"
                )
            lib_path = str(found)

        self._lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        self._initialized = False

    def _setup_functions(self):
        """Declare function signatures for type safety."""
        # void prlx_init(void)
        self._lib.prlx_init.restype = None
        self._lib.prlx_init.argtypes = []

        # void prlx_pre_launch(const char* kernel_name, dim3 grid, dim3 block)
        # dim3 is passed as 3 uint32_t values in CUDA, but the C function
        # takes dim3 structs. We'll use a wrapper approach.

        # void prlx_post_launch(void)
        self._lib.prlx_post_launch.restype = None
        self._lib.prlx_post_launch.argtypes = []

        # void prlx_shutdown(void)
        self._lib.prlx_shutdown.restype = None
        self._lib.prlx_shutdown.argtypes = []

        # void prlx_session_begin(const char* name)
        self._lib.prlx_session_begin.restype = None
        self._lib.prlx_session_begin.argtypes = [ctypes.c_char_p]

        # void prlx_session_end(void)
        self._lib.prlx_session_end.restype = None
        self._lib.prlx_session_end.argtypes = []

    def init(self):
        """Initialize the tracing system. Reads PRLX_* environment variables."""
        self._lib.prlx_init()
        self._initialized = True

    def pre_launch(
        self,
        kernel_name: str,
        grid_dim: Tuple[int, int, int],
        block_dim: Tuple[int, int, int],
    ):
        """
        Set up trace buffer before a kernel launch.

        Args:
            kernel_name: Name of the CUDA kernel.
            grid_dim: Grid dimensions (x, y, z).
            block_dim: Block dimensions (x, y, z).
        """
        if not self._initialized:
            self.init()

        # dim3 is a struct { unsigned int x, y, z; } -- 12 bytes
        class Dim3(ctypes.Structure):
            _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

        # Set up the pre_launch function with dim3 args
        self._lib.prlx_pre_launch.restype = None
        self._lib.prlx_pre_launch.argtypes = [
            ctypes.c_char_p,
            Dim3,
            Dim3,
        ]

        grid = Dim3(grid_dim[0], grid_dim[1], grid_dim[2])
        block = Dim3(block_dim[0], block_dim[1], block_dim[2])

        self._lib.prlx_pre_launch(kernel_name.encode("utf-8"), grid, block)

    def post_launch(self):
        """Copy trace buffer from device and write to file."""
        self._lib.prlx_post_launch()

    def session_begin(self, name: str):
        """Begin a session (multi-kernel tracing). Creates a directory for traces."""
        if not self._initialized:
            self.init()
        self._lib.prlx_session_begin(name.encode("utf-8"))

    def session_end(self):
        """End the current session and write the manifest."""
        self._lib.prlx_session_end()

    def shutdown(self):
        """Clean up resources."""
        self._lib.prlx_shutdown()
        self._initialized = False

    def __del__(self):
        if self._initialized:
            try:
                self.shutdown()
            except Exception:
                pass


@contextmanager
def tracing(
    kernel_name: str,
    grid_dim: Tuple[int, int, int],
    block_dim: Tuple[int, int, int],
    output: Optional[str] = None,
    history_depth: Optional[int] = None,
    lib_path: Optional[str] = None,
):
    """
    Context manager for automatic tracing.

    Usage:
        with prlx.runtime.tracing("my_kernel", (1,1,1), (256,1,1)):
            # launch kernel
            pass
        # trace file is written automatically

    Args:
        kernel_name: Name of the CUDA kernel.
        grid_dim: Grid dimensions.
        block_dim: Block dimensions.
        output: Output trace file path (default: PRLX_TRACE env or "trace.prlx").
        history_depth: History ring depth (0 to disable, default from env).
        lib_path: Explicit path to shared library.
    """
    if output:
        os.environ["PRLX_TRACE"] = output
    if history_depth is not None:
        os.environ["PRLX_HISTORY_DEPTH"] = str(history_depth)

    rt = PrlxRuntime(lib_path)
    rt.pre_launch(kernel_name, grid_dim, block_dim)
    try:
        yield rt
    finally:
        rt.post_launch()


@contextmanager
def session(name: str, lib_path: Optional[str] = None):
    """
    Context manager for session-based multi-kernel tracing.

    Usage:
        with prlx.runtime.session("my_pipeline"):
            # launch multiple kernels — each gets its own trace file
            pass
        # session.json manifest is written automatically

    Args:
        name: Session directory name (will be created).
        lib_path: Explicit path to shared library.
    """
    rt = PrlxRuntime(lib_path)
    rt.session_begin(name)
    try:
        yield rt
    finally:
        rt.session_end()
