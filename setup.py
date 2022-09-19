import os
import re
import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "The preferred way to invoke 'setup.py' is via pip, as in 'pip "
        "install .'. If you wish to run the setup script directly, you must "
        "first install the build dependencies listed in pyproject.toml!",
        file=sys.stderr,
    )
    raise

setup(
    name="nanogram",
    version="0.0.1",
    license="BSD",
    packages=["nanogram"],
    package_dir={"": "src"},
    cmake_install_dir="src/nanogram",
    include_package_data=True,
    python_requires=">=3.8",
)
