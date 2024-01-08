"""Indices."""

import typing

from basix._basixcpp import index as _index

__all__: typing.List[str] = []


def index(p: int, q: typing.Optional[int] = None, r: typing.Optional[int] = None) -> int:
    """TODO."""
    if q is None:
        assert r is None
        return _index(p)
    elif r is None:
        return _index(p, q)
    else:
        return _index(p, q, r)
