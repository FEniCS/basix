"""Functions to manipulate variant types."""

from ._basixcpp import LagrangeVariant as _LV
from ._basixcpp import DPCVariant as _DV


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
    if variant.lower() == "gll":
        return _LV.gll_warped
    if variant.lower() == "chebyshev":
        return _LV.chebyshev_isaac
    if variant.lower() == "gl":
        return _LV.gl_isaac

    if not hasattr(_LV, variant.lower()):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_LV, variant.lower())


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


def string_to_dpc_variant(variant: str) -> _DV:
    """
    Convert a string to a Basix DPCVariant enum.

    Parameters
    ----------
    variant : str
        The DPC variant as a string.

    Returns
    -------
    basix.DPCVariant
        The DPC variant
    """
    if not hasattr(_DV, variant.lower()):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_DV, variant.lower())


def dpc_variant_to_string(variant: _DV) -> str:
    """
    Convert a Basix DPCVariant enum to a string.

    Parameters
    ----------
    variant : basix.DPCVariant
        The DPC variant

    Returns
    -------
    str
        The DPC variant as a string.
    """
    return variant.name
