"""Centralized logging for the prlx package.

Usage::

    from prlx._log import get_logger
    logger = get_logger(__name__)   # e.g. "prlx.triton_hook"
    logger.info("Hooked Triton stages API")

The root ``prlx`` logger defaults to WARNING on stderr.
Set ``PRLX_LOG_LEVEL`` (DEBUG/INFO/WARNING/ERROR) to override.
"""

import logging
import os

_ROOT_LOGGER_NAME = "prlx"
_configured = False


def _configure_root() -> logging.Logger:
    global _configured
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if _configured:
        return root
    _configured = True

    level_name = os.environ.get("PRLX_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = "[prlx] %(name)s:%(lineno)d %(message)s" if level <= logging.DEBUG else "[prlx] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``prlx`` hierarchy.

    If *name* already starts with ``prlx``, it is used as-is;
    otherwise ``prlx.`` is prepended.
    """
    _configure_root()
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
