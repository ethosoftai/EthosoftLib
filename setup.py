"""Mark wheels with a staged native payload as platform-specific (never universal)."""
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class BinaryAwareWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        root = Path(__file__).resolve().parent / "src/ethosoftlib"
        if any(path.is_file() for family in ("mercan", "nedo")
               for ext in ("*.so", "*.dylib", "*.dll")
               for path in (root / family / "lib").glob(ext)):
            self.root_is_pure = False


setup(cmdclass={"bdist_wheel": BinaryAwareWheel})
