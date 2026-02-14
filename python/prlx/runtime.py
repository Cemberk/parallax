"""ctypes FFI wrapper around libprlx_runtime_shared.so."""

import ctypes
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

from ._find_lib import find_runtime_library


class Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class PrlxRuntime:

    def __init__(self, lib_path: Optional[str] = None):
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
        self._lib.prlx_init.restype = None
        self._lib.prlx_init.argtypes = []

        self._lib.prlx_pre_launch.restype = None
        self._lib.prlx_pre_launch.argtypes = [ctypes.c_char_p, Dim3, Dim3]

        self._lib.prlx_post_launch.restype = None
        self._lib.prlx_post_launch.argtypes = []

        self._lib.prlx_shutdown.restype = None
        self._lib.prlx_shutdown.argtypes = []

        self._lib.prlx_session_begin.restype = None
        self._lib.prlx_session_begin.argtypes = [ctypes.c_char_p]

        self._lib.prlx_session_end.restype = None
        self._lib.prlx_session_end.argtypes = []

    def init(self):
        self._lib.prlx_init()
        self._initialized = True

    def pre_launch(self, kernel_name: str, grid_dim: Tuple[int, ...], block_dim: Tuple[int, ...]):
        if not self._initialized:
            self.init()

        grid = Dim3(grid_dim[0], grid_dim[1], grid_dim[2])
        block = Dim3(block_dim[0], block_dim[1], block_dim[2])
        self._lib.prlx_pre_launch(kernel_name.encode("utf-8"), grid, block)

    def post_launch(self):
        self._lib.prlx_post_launch()

    def session_begin(self, name: str):
        if not self._initialized:
            self.init()
        self._lib.prlx_session_begin(name.encode("utf-8"))

    def session_end(self):
        self._lib.prlx_session_end()

    def shutdown(self):
        self._lib.prlx_shutdown()
        self._initialized = False

    def __del__(self):
        if self._initialized:
            self.shutdown()


@contextmanager
def tracing(kernel_name, grid_dim, block_dim, output=None, history_depth=None, lib_path=None):
    """Context manager: sets up tracing around a kernel launch."""
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
    """Context manager: multi-kernel session tracing."""
    rt = PrlxRuntime(lib_path)
    rt.session_begin(name)
    try:
        yield rt
    finally:
        rt.session_end()
