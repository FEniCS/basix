import os
import pytest

# Get all the demos in this folder
path = os.path.dirname(os.path.realpath(__file__))
demos = []
for folder in os.listdir(path):
    if folder.startswith("demo_"):
        subpath = os.path.join(path, folder)
        if os.path.isdir(subpath) and os.path.isfile(os.path.join(subpath, "main.cpp")):
            demos.append(folder)


@pytest.mark.parametrize("demo", demos)
def test_demo(demo):
    demo_build = f"{path}/{demo}/_build"
    assert os.system(f"mkdir -p {demo_build} && cd {demo_build} && cmake .. && make && ./{demo}") == 0
