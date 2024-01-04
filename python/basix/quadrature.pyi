import numpy as _np
import numpy.typing as _npt
import typing as _typing
from basix._basixcpp import CellType as _CT, PolysetType as _PT, QuadratureType as _QT

__all__ = ['string_to_type', 'type_to_string', 'make_quadrature']

def string_to_type(rule: str) -> _QT: ...
def type_to_string(quadraturetype: _QT) -> str: ...
def make_quadrature(cell: _CT, degree: int, rule: _QT = ..., polyset_type: _PT = ...) -> _typing.Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]: ...
