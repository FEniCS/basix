"""Functions for handling Sobolev spaces."""

import typing as _typing

from ._basixcpp import SobolevSpace as _SS
from ._basixcpp import sobolev_space_intersection as _ssi


def intersection(spaces: _typing.List[_SS]) -> _SS:
    """Compute the intersection of a list of Sobolev spaces.

    Args:
      spaces: A list of Sobolev spaces.

    Returns:
      The intersection of the Sobolev spaces.

    """
    space = spaces[0]
    for s in spaces[1:]:
        space = _ssi(space, s)
    return space
