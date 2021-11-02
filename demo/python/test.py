import os
import pytest
import subprocess
import sys

# Get all the demos in this folder
path = os.path.dirname(os.path.realpath(__file__))
demos = []
for file in os.listdir(path):
    if file.endswith(".py") and file.startswith("demo"):
        demos.append(file)


@pytest.mark.parametrize("demo", demos)
def test_demo(demo):
    ret = subprocess.run([sys.executable, demo], cwd=str(path), check=True)
    assert ret.returncode == 0
