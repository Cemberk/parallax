"""Tests for PyTorch integration (prlx.pytorch_hook).

All tests are pure mock tests -- NO GPU, NO torch import required.
We mock everything that would require torch or real hardware.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


def _make_fake_path(path_str, exists=True):
    """Create a mock that behaves like a Path with controllable .exists()."""
    p = mock.MagicMock(spec=Path)
    p.exists.return_value = exists
    p.__str__ = mock.MagicMock(return_value=path_str)
    p.__fspath__ = mock.MagicMock(return_value=path_str)
    p.parent = Path(path_str).parent
    return p


class TestPytorchHookImportable:
    def test_pytorch_hook_importable(self):
        """Module imports without torch installed.

        The pytorch_hook module imports ``from ._find_lib import ...``
        at module level, which does NOT require torch.  The actual
        ``import torch`` only happens inside ``install()`` and friends.
        """
        import prlx.pytorch_hook
        assert hasattr(prlx.pytorch_hook, "install")
        assert hasattr(prlx.pytorch_hook, "uninstall")
        assert hasattr(prlx.pytorch_hook, "PrlxTorchWrapper")


class TestEnablePytorchCallable:
    def test_enable_pytorch_callable(self):
        """prlx.enable_pytorch exists and is callable."""
        import prlx
        assert hasattr(prlx, "enable_pytorch")
        assert callable(prlx.enable_pytorch)


class TestPytorchTraceCallable:
    def test_pytorch_trace_callable(self):
        """prlx.pytorch_trace exists and is callable."""
        import prlx
        assert hasattr(prlx, "pytorch_trace")
        assert callable(prlx.pytorch_trace)


class TestCppExtensionHookAddsFlag:
    def test_cpp_extension_hook_adds_flag(self):
        """Mock torch.utils.cpp_extension.load_inline, call install,
        verify pass plugin flag is injected into extra_cuda_cflags."""
        import prlx.pytorch_hook as phook

        # Reset module-level state so install() doesn't short-circuit
        phook._installed = False
        phook._original_load_inline = None
        phook._tier1_active = False
        phook._tier2_active = False
        phook._tier3_active = False

        fake_pass = _make_fake_path("/fake/lib/libPrlxPass.so")
        fake_runtime = _make_fake_path("/fake/lib/libprlx_runtime_shared.so")

        # Build a fake torch.utils.cpp_extension module
        fake_cpp_ext = SimpleNamespace(load_inline=lambda *a, **kw: "original_result")

        # Install the fake torch modules into sys.modules
        fake_torch = SimpleNamespace(
            utils=SimpleNamespace(
                cpp_extension=fake_cpp_ext,
            ),
        )
        modules_to_inject = {
            "torch": fake_torch,
            "torch.utils": fake_torch.utils,
            "torch.utils.cpp_extension": fake_cpp_ext,
        }

        with mock.patch.dict(sys.modules, modules_to_inject):
            with mock.patch.object(phook, "find_pass_plugin", return_value=fake_pass):
                with mock.patch.object(phook, "find_runtime_library", return_value=fake_runtime):
                    # Call install with only Tier 2
                    phook.install(
                        instrument_triton=False,
                        instrument_extensions=True,
                        nvbit_precompiled=False,
                        verbose=False,
                    )

                    # Now load_inline should be patched
                    assert phook._tier2_active is True
                    assert phook._original_load_inline is not None

                    # Call the patched load_inline and verify flags
                    result_kwargs = {}

                    def capture_original(*args, **kwargs):
                        result_kwargs.update(kwargs)
                        return "patched_result"

                    phook._original_load_inline = capture_original

                    patched = fake_cpp_ext.load_inline
                    ret = patched(name="test", extra_cuda_cflags=[])

                    cuda_cflags = result_kwargs.get("extra_cuda_cflags", [])
                    ldflags = result_kwargs.get("extra_ldflags", [])
                    assert any("-fpass-plugin=" in f for f in cuda_cflags), \
                        f"Expected -fpass-plugin flag in {cuda_cflags}"
                    assert any("-lprlx_runtime_shared" in f for f in ldflags), \
                        f"Expected -lprlx_runtime_shared in {ldflags}"

        # Cleanup
        phook._installed = False
        phook._original_load_inline = None
        phook._tier2_active = False


class TestNvbitEnvSetup:
    def test_nvbit_env_setup(self):
        """Mock everything, call _setup_nvbit, verify LD_PRELOAD is set."""
        import prlx.pytorch_hook as phook

        # Reset state
        phook._tier3_active = False

        fake_nvbit = _make_fake_path("/fake/lib/libprlx_nvbit.so")

        # Save and clear LD_PRELOAD
        orig_preload = os.environ.get("LD_PRELOAD")
        if "LD_PRELOAD" in os.environ:
            del os.environ["LD_PRELOAD"]

        try:
            with mock.patch.object(phook, "find_nvbit_library", return_value=fake_nvbit):
                phook._setup_nvbit(verbose=False)

                assert phook._tier3_active is True
                assert "LD_PRELOAD" in os.environ
                assert "/fake/lib/libprlx_nvbit.so" in os.environ["LD_PRELOAD"]
        finally:
            # Restore LD_PRELOAD
            if orig_preload is not None:
                os.environ["LD_PRELOAD"] = orig_preload
            elif "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
            phook._tier3_active = False

    def test_nvbit_env_appends_to_existing(self):
        """When LD_PRELOAD already has a value, _setup_nvbit prepends to it."""
        import prlx.pytorch_hook as phook

        phook._tier3_active = False

        fake_nvbit = _make_fake_path("/fake/lib/libprlx_nvbit.so")

        orig_preload = os.environ.get("LD_PRELOAD")
        os.environ["LD_PRELOAD"] = "/existing/lib.so"

        try:
            with mock.patch.object(phook, "find_nvbit_library", return_value=fake_nvbit):
                phook._setup_nvbit(verbose=False)

                assert "/fake/lib/libprlx_nvbit.so" in os.environ["LD_PRELOAD"]
                assert "/existing/lib.so" in os.environ["LD_PRELOAD"]
        finally:
            if orig_preload is not None:
                os.environ["LD_PRELOAD"] = orig_preload
            elif "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
            phook._tier3_active = False


class TestUninstallRestoresState:
    def test_uninstall_restores_state(self):
        """Call install then uninstall, verify state is restored."""
        import prlx.pytorch_hook as phook

        # Reset module-level state
        phook._installed = False
        phook._original_load_inline = None
        phook._tier1_active = False
        phook._tier2_active = False
        phook._tier3_active = False

        fake_pass = _make_fake_path("/fake/lib/libPrlxPass.so")
        fake_runtime = _make_fake_path("/fake/lib/libprlx_runtime_shared.so")
        fake_nvbit = _make_fake_path("/fake/lib/libprlx_nvbit.so")

        # Build fake torch module hierarchy
        original_load_inline_fn = lambda *a, **kw: "original"
        fake_cpp_ext = SimpleNamespace(load_inline=original_load_inline_fn)
        fake_torch = SimpleNamespace(
            utils=SimpleNamespace(cpp_extension=fake_cpp_ext),
        )
        modules_to_inject = {
            "torch": fake_torch,
            "torch.utils": fake_torch.utils,
            "torch.utils.cpp_extension": fake_cpp_ext,
        }

        orig_preload = os.environ.get("LD_PRELOAD")
        if "LD_PRELOAD" in os.environ:
            del os.environ["LD_PRELOAD"]

        try:
            with mock.patch.dict(sys.modules, modules_to_inject):
                with mock.patch.object(phook, "find_pass_plugin", return_value=fake_pass):
                    with mock.patch.object(phook, "find_runtime_library", return_value=fake_runtime):
                        with mock.patch.object(phook, "find_nvbit_library", return_value=fake_nvbit):
                            # Install Tier 2 and Tier 3 (skip Tier 1 which needs real Triton)
                            phook.install(
                                instrument_triton=False,
                                instrument_extensions=True,
                                nvbit_precompiled=True,
                                verbose=False,
                            )

                            assert phook._installed is True
                            assert phook._tier2_active is True
                            assert phook._tier3_active is True
                            assert "LD_PRELOAD" in os.environ

                            # Now uninstall
                            phook.uninstall()

                            assert phook._installed is False
                            assert phook._tier2_active is False
                            assert phook._tier3_active is False
                            assert phook._original_load_inline is None

                            # load_inline should be restored
                            assert fake_cpp_ext.load_inline is original_load_inline_fn

                            # LD_PRELOAD should be cleaned
                            ld_preload = os.environ.get("LD_PRELOAD", "")
                            assert "/fake/lib/libprlx_nvbit.so" not in ld_preload
        finally:
            if orig_preload is not None:
                os.environ["LD_PRELOAD"] = orig_preload
            elif "LD_PRELOAD" in os.environ:
                del os.environ["LD_PRELOAD"]
            phook._installed = False
            phook._original_load_inline = None
            phook._tier1_active = False
            phook._tier2_active = False
            phook._tier3_active = False


class TestPrlxTorchWrapper:
    def test_wrapper_sets_env_variables(self):
        """PrlxTorchWrapper should set PRLX_SESSION on enter and clean up on exit."""
        from prlx.pytorch_hook import PrlxTorchWrapper

        orig_session = os.environ.get("PRLX_SESSION")

        try:
            with mock.patch("prlx.pytorch_hook.find_runtime_library", return_value=None):
                wrapper = PrlxTorchWrapper("my_test_session", output="/tmp/test_trace")
                with wrapper:
                    assert os.environ.get("PRLX_SESSION") == "my_test_session"
                    assert os.environ.get("PRLX_TRACE") == "/tmp/test_trace"

                # After exit, env vars should be cleaned up
                assert os.environ.get("PRLX_SESSION") != "my_test_session" or "PRLX_SESSION" not in os.environ
        finally:
            if orig_session is not None:
                os.environ["PRLX_SESSION"] = orig_session
            elif "PRLX_SESSION" in os.environ:
                del os.environ["PRLX_SESSION"]
            if "PRLX_TRACE" in os.environ:
                del os.environ["PRLX_TRACE"]


class TestCmdPytorchInfo:
    def test_cmd_pytorch_info(self, capsys):
        """prlx pytorch --info should print integration status."""
        from prlx.cli import cmd_pytorch

        args = SimpleNamespace(
            info=True,
            nvbit=False,
            output=None,
            script=None,
            script_args=[],
        )

        with mock.patch("prlx.cli._find_lib") as mock_find:
            mock_find.find_pass_plugin.return_value = None
            mock_find.find_differ_binary.return_value = None
            with mock.patch("prlx.cli._find_nvbit_tool", return_value=None):
                result = cmd_pytorch(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "PRLX PyTorch Integration" in captured.out
