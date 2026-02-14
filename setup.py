"""Minimal setup.py to force platform-specific wheel tag.

The prlx package bundles pre-built native binaries (.so, ELF) as package
data, so wheels must be tagged with the target platform (e.g.,
manylinux_2_28_x86_64) instead of the default 'py3-none-any'.

Because the binaries are Python-version-independent, we tag as
'py3-none-<platform>' so a single wheel covers all Python 3.x versions.
"""
from setuptools import setup
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            # Keep the platform tag from the parent, override python/abi
            _, _, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass = {"bdist_wheel": bdist_wheel}
except ImportError:
    cmdclass = {}


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution, cmdclass=cmdclass)
