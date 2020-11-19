import libtab
import os


def test_include():
    path_to_file = os.path.join(libtab.get_include_path(), "libtab.h")
    assert os.path.isfile(path_to_file)
