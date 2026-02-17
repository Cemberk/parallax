"""Thin ctypes wrapper around HIP Runtime API functions needed for module binding.

Mirrors _cuda_driver.py for AMD ROCm / HIP platforms.
Used by _module_binder.py when running on AMD GPUs.
"""

import ctypes
from typing import Tuple

try:
    _libhip = ctypes.CDLL("libamdhip64.so")
except OSError:
    _libhip = None

HIP_SUCCESS = 0


class HIPDriverError(RuntimeError):
    """Raised when a HIP API call fails."""
    pass


def _check(result: int, func_name: str):
    if result != HIP_SUCCESS:
        raise HIPDriverError(f"{func_name} failed with error code {result}")


def available() -> bool:
    """Return True if the HIP runtime library is loaded."""
    return _libhip is not None


def hipModuleGetGlobal(hip_module: int, name: str) -> Tuple[int, int]:
    """Get the device pointer and size of a global variable in a HIP module.

    Args:
        hip_module: hipModule_t handle.
        name: Name of the global variable (e.g. "g_prlx_buffer").

    Returns:
        (device_ptr, size_bytes) tuple.
    """
    dptr = ctypes.c_uint64(0)
    size = ctypes.c_size_t(0)
    result = _libhip.hipModuleGetGlobal(
        ctypes.byref(dptr),
        ctypes.byref(size),
        ctypes.c_void_p(hip_module),
        name.encode("utf-8"),
    )
    _check(result, f"hipModuleGetGlobal({name})")
    return dptr.value, size.value


def hipMemcpyHtoD(dst_device: int, src_host: ctypes.c_void_p, size: int):
    """Copy data from host to device.

    Args:
        dst_device: Device pointer (hipDeviceptr_t).
        src_host: Pointer to host data.
        size: Number of bytes to copy.
    """
    result = _libhip.hipMemcpyHtoD(
        ctypes.c_uint64(dst_device),
        src_host,
        ctypes.c_size_t(size),
    )
    _check(result, "hipMemcpyHtoD")


def write_device_global_ptr(hip_module: int, name: str, device_ptr: int):
    """Write a device pointer value into a module's global variable.

    Args:
        hip_module: hipModule_t handle.
        name: Name of the device global (e.g. "g_prlx_buffer").
        device_ptr: The device pointer value to write.
    """
    dptr, _size = hipModuleGetGlobal(hip_module, name)
    val = ctypes.c_uint64(device_ptr)
    hipMemcpyHtoD(dptr, ctypes.byref(val), 8)


def write_device_global_u32(hip_module: int, name: str, value: int):
    """Write a uint32 value into a module's global variable.

    Args:
        hip_module: hipModule_t handle.
        name: Name of the device global (e.g. "__prlx_sample_rate").
        value: The uint32 value to write.
    """
    dptr, _size = hipModuleGetGlobal(hip_module, name)
    val = ctypes.c_uint32(value)
    hipMemcpyHtoD(dptr, ctypes.byref(val), 4)
