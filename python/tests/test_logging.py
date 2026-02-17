"""Tests for prlx._log centralized logging."""

import logging
import os

import pytest


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset prlx logging state between tests."""
    import prlx._log as _log
    _log._configured = False
    root = logging.getLogger("prlx")
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    _log._configured = False
    root.handlers.clear()


def test_default_level_is_warning():
    os.environ.pop("PRLX_LOG_LEVEL", None)
    from prlx._log import get_logger
    logger = get_logger("prlx.test")
    assert logging.getLogger("prlx").level == logging.WARNING


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("PRLX_LOG_LEVEL", "DEBUG")
    from prlx._log import get_logger
    logger = get_logger("prlx.test")
    assert logging.getLogger("prlx").level == logging.DEBUG


def test_stderr_output(monkeypatch, capsys):
    monkeypatch.setenv("PRLX_LOG_LEVEL", "INFO")
    from prlx._log import get_logger
    logger = get_logger("prlx.test_stderr")
    logger.info("hello from test")
    captured = capsys.readouterr()
    assert "hello from test" in captured.err


def test_warning_hides_info(monkeypatch, capsys):
    monkeypatch.delenv("PRLX_LOG_LEVEL", raising=False)
    from prlx._log import get_logger
    logger = get_logger("prlx.test_suppress")
    logger.info("should not appear")
    captured = capsys.readouterr()
    assert "should not appear" not in captured.err


def test_get_logger_prefixes_name():
    from prlx._log import get_logger
    logger = get_logger("triton_hook")
    assert logger.name == "prlx.triton_hook"


def test_get_logger_keeps_full_name():
    from prlx._log import get_logger
    logger = get_logger("prlx.pytorch_hook")
    assert logger.name == "prlx.pytorch_hook"
