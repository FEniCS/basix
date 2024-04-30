"""Script for testing."""

import os
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
    run(f"cmake {cmake_args} -B {demo_build} -S {demo_source}", check=True, shell=True)
    run(f"cmake --build {demo_build}", check=True, shell=True)
    run(os.path.join(demo_build, demo), check=True, shell=True)
