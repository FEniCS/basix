from .libtab import *
from .libtab import __version__
import os
_include_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

def get_include_path():
    return _include_path
