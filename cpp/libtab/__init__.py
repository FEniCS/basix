from .libtab import *
from .libtab import __version__
import os
_prefix_dir = os.path.dirname(os.path.abspath(__file__))

def get_include_path():
    return os.path.join(_prefix_dir, "include")

def get_prefix_dir():
    return _prefix_dir
