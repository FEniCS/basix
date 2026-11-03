import importlib.util
import os
import subprocess
import sys

import pytest

# Get all the demos in this folder
path = os.path.dirname(os.path.realpath(__file__))
demos = []
for file in os.listdir(path):
    if file.endswith(".py") and file.startswith("demo"):
        demos.append(file)


@pytest.mark.parametrize("demo", demos)
def test_demo(demo):
    if demo == "demo_numba_kernels.py":
        # Numba is not available on all platforms that Basix supports.
        # Locate it without importing it: the demo runs in a subprocess,
        # so importing Numba here would only load its shared libraries
        # into the process that runs under valgrind.
        if importlib.util.find_spec("numba") is None:
            pytest.skip("Numba is not available")
    ret = subprocess.run([sys.executable, demo], cwd=str(path), check=True)
    assert ret.returncode == 0
