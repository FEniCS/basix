"""Functions to manipulate variant types."""

from ._basixcpp import LagrangeVariant as _LV


def string_to_lagrange_variant(variant: str) -> _LV:
    """
    Convert a string to a Basix LagrangeVariant enum.

    Parameters
    ----------
    variant : str
        The Lagrange variant as a string.

    Returns
    -------
    basix.LagrangeVariant
        The Lagrange variant
    """
    if variant == "gll":
        return _LV.gll_warped
    if variant == "chebyshev":
        return _LV.chebyshev_isaac
    if variant == "gl":
        return _LV.gl_isaac

    if not hasattr(_LV, variant):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_LV, variant)


def lagrange_variant_to_string(variant: _LV) -> str:
    """
    Convert a Basix LagrangeVariant enum to a string.

    Parameters
    ----------
    variant : basix.LagrangeVariant
        The Lagrange variant

    Returns
    -------
    str
        The Lagrange variant as a string.
    """
    return variant.name
