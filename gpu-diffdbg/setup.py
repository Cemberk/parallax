"""Minimal setup.py to force platform-specific wheel tag.

The prlx package bundles pre-built native binaries (.so, ELF) as package
data, so wheels must be tagged with the target platform (e.g.,
manylinux_2_28_x86_64) instead of the default 'py3-none-any'.
"""
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
