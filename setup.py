"""Mark wheels with a staged native payload as platform-specific (never universal)."""
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class BinaryAwareWheel(bdist_wheel):
    def get_tag(self):
        python_tag, abi_tag, platform_tag = super().get_tag()
        # Bundled libmercan/NedoTokenizer are loaded via ctypes and do not
        # reference the CPython ABI. One OS/arch wheel supports Python >= 3.10.
        if not self.root_is_pure:
            return "py3", "none", platform_tag
        return python_tag, abi_tag, platform_tag

    def finalize_options(self):
        super().finalize_options()
        root = Path(__file__).resolve().parent / "src/ethosoftlib"
        if any(path.is_file() for family in ("mercan", "nedo")
               for ext in ("*.so", "*.dylib", "*.dll")
               for path in (root / family / "lib").glob(ext)):
            self.root_is_pure = False


setup(cmdclass={"bdist_wheel": BinaryAwareWheel})
