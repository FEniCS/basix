"""Script for testing."""

import os
import sys
from subprocess import run

import pytest

# Get all the demos in this folder
path = os.path.dirname(os.path.realpath(__file__))
demos = []
for folder in os.listdir(path):
    if folder.startswith("demo_"):
        subpath = os.path.join(path, folder)
        if os.path.isdir(subpath) and os.path.isfile(os.path.join(subpath, "main.cpp")):
            demos.append(folder)


@pytest.fixture
def cmake_args(request):
    return request.config.getoption("--cmake-args")


@pytest.mark.parametrize("demo", demos)
def test_demo(demo, cmake_args):
    """Test demos."""
    demo_source = os.path.join(path, demo)
    demo_build = os.path.join(path, demo, "_build")

    # There is no cross-platform way to get cmake to execute a target.
    # See e.g. https://discourse.cmake.org/t/feature-request-cmake-run-target/9170
    # TODO: Check existence of cross-platform target executer in cmake.
    if sys.platform.startswith("win32"):
        # Assume default generator is MSVC generator with multiple build targets
        run(f"cmake {cmake_args} -B {demo_build} -S {demo_source}", check=True, shell=True)
        # MSVC produces really slow binaries with --config Debug.
        run(f"cmake --build {demo_build} --config Release", check=True, shell=True)

        # MSVC generator supports multiple build targets per cmake configuration, each gets
        # its own subdirectory in {demo_build} e.g. Debug/ Release/ etc.
        demo_executable = demo + ".exe"
        run(os.path.join(demo_build, "Release", demo_executable), check=True, shell=True)
    else:
        # Uses default generator (usually make)
        run(
            f"cmake -DCMAKE_BUILD_TYPE=Debug {cmake_args} -B {demo_build} -S {demo_source}",
            check=True,
            shell=True,
        )
        run(f"cmake --build {demo_build}", check=True, shell=True)

        demo_executable = demo
        run(os.path.join(demo_build, demo_executable), check=True, shell=True)
