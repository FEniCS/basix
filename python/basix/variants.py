"""Functions to manipulate variant types."""

from ._basixcpp import LagrangeVariant as _LV
from ._basixcpp import DPCVariant as _DV


def string_to_lagrange_variant(variant: str) -> _LV:
    """Convert a string to a Basix LagrangeVariant enum.

    Args:
        variant: The Lagrange variant as a string.

    Returns:
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
    """Convert a Basix LagrangeVariant enum to a string.

    Args:
        variant: The Lagrange variant

    Returns:
        The Lagrange variant as a string.
    """
    return variant.name


def string_to_dpc_variant(variant: str) -> _DV:
    """Convert a string to a Basix DPCVariant enum.

    Args:
        variant: The DPC variant as a string.

    Returns:
        The DPC variant
    """
    if not hasattr(_DV, variant.lower()):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_DV, variant.lower())


def dpc_variant_to_string(variant: _DV) -> str:
    """Convert a Basix DPCVariant enum to a string.

    Args:
        variant: The DPC variant

    Returns:
        The DPC variant as a string.
    """
    return variant.name
