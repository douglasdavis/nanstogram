from skbuild import setup


setup(
    name="nanstogram",
    version="0.0.1",
    license="BSD",
    packages=["nanstogram"],
    package_dir={"": "src"},
    cmake_install_dir="src/nanstogram",
    include_package_data=True,
    python_requires=">=3.8",
)
