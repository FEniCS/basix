import numpy as _np
import numpy.typing as npt
from basix._basixcpp import CellType as _CT, PolynomialType as _PT

__all__ = ['reshape_coefficients']

def reshape_coefficients(poly_type: _PT, cell_type: _CT, coefficients: npt.NDArray[_np.float64], value_size: int, input_degree: int, output_degree: int) -> npt.NDArray[_np.float64]: ...
