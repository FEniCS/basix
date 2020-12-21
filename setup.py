from skbuild import setup

setup(
    name="fenics-basix",
    version="0.0.1",
    description="FEniCS tabulation library",
    author="FEniCS Project",
    license="MIT",
    packages=["basix"],
    package_dir={"": "cpp"},
    cmake_install_dir="cpp/basix",
)
