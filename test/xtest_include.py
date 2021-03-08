import basix
import os


def test_include():
    path_to_file = os.path.join(basix.get_include_path(), "basix.h")
    assert os.path.isfile(path_to_file)
