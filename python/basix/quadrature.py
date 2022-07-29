"""Functions to manipulate quadrature types."""

from ._basixcpp import QuadratureType as _QT


def string_to_type(rule: str) -> _QT:
    """Convert a string to a Basix QuadratureType enum.

    Args:
        rule: The quadrature rule as a string.

    Returns:
        The quadrature type
    """
    if rule == "default":
        return _QT.Default

    if rule in ["Gauss-Lobatto-Legendre", "GLL"]:
        return _QT.gll
    if rule in ["Gauss-Legendre", "GL", "Gauss-Jacobi"]:
        return _QT.gauss_jacobi
    if rule == "Xiao-Gimbutas":
        return _QT.xiao_gimbutas

    if not hasattr(_QT, rule):
        raise ValueError(f"Unknown quadrature rule: {rule}")
    return getattr(_QT, rule)


def type_to_string(quadraturetype: _QT) -> str:
    """Convert a Basix QuadratureType enum to a string.

    Args:
        quadraturetype: The quadrature type

    Returns:
        The quadrature rule as a string.
    """
    return quadraturetype.name
