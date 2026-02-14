"""Binds PRLX device globals into Triton JIT-compiled CUmodules.

The core problem: cudaMemcpyToSymbol in prlx_host.cu writes device globals
(g_prlx_buffer, etc.) into the shared library's CUDA module, but Triton kernels
are loaded into separate CUmodules via cuModuleLoadData. Their copies of these
globals stay nullptr, so recording functions bail out and capture 0 events.

This module uses the CUDA Driver API to write the correct buffer pointers into
each Triton kernel's CUmodule before launch.
"""

import sys
from typing import Set

from . import _cuda_driver as cu
from .runtime import PrlxRuntime


class ModuleBinder:
    """Binds PRLX runtime state into Triton kernel CUmodules."""

    def __init__(self, runtime: PrlxRuntime, verbose: bool = False):
        self._runtime = runtime
        self._verbose = verbose
        self._bound_modules: Set[int] = set()

    def invalidate(self):
        """Clear the bound-modules cache.

        Must be called after pre_launch() because buffer pointers change
        between launches (freed in post_launch, reallocated in pre_launch).
        """
        self._bound_modules.clear()

    def bind_module(self, cu_function: int):
        """Write all PRLX device globals into the CUmodule owning cu_function.

        Args:
            cu_function: CUfunction handle from Triton's launch_enter_hook metadata.
        """
        if not cu.available():
            return

        cu_module = cu.cuFuncGetModule(cu_function)
        if cu_module in self._bound_modules:
            return

        trace_buf = self._runtime.get_trace_buffer()
        if trace_buf == 0:
            # pre_launch hasn't run yet or buffers not allocated
            return

        if self._verbose:
            print(f"[prlx] Binding device globals to CUmodule 0x{cu_module:x}", file=sys.stderr)

        # Pointer globals (8 bytes each)
        cu.write_device_global_ptr(cu_module, "g_prlx_buffer", trace_buf)

        history_buf = self._runtime.get_history_buffer()
        cu.write_device_global_ptr(cu_module, "g_prlx_history_buffer", history_buf)

        snapshot_buf = self._runtime.get_snapshot_buffer()
        cu.write_device_global_ptr(cu_module, "g_prlx_snapshot_buffer", snapshot_buf)

        # uint32 globals (4 bytes each)
        cu.write_device_global_u32(cu_module, "__prlx_sample_rate", self._runtime.get_sample_rate())
        cu.write_device_global_u32(cu_module, "g_prlx_history_depth", self._runtime.get_history_depth())
        cu.write_device_global_u32(cu_module, "g_prlx_snapshot_depth", self._runtime.get_snapshot_depth())

        self._bound_modules.add(cu_module)
