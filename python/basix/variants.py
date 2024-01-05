"""Functions to manipulate variant types."""

from basix._basixcpp import DPCVariant as _DV
from basix._basixcpp import LagrangeVariant as _LV

__all__ = ["string_to_lagrange_variant", "lagrange_variant_to_string",
           "string_to_dpc_variant", "dpc_variant_to_string"]


def string_to_lagrange_variant(variant: str) -> _LV:
    """Convert a string to a Basix LagrangeVariant enum.

    Args:
        variant: Lagrange variant as a string.

    Returns:
        The Lagrange variant.

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
        variant: Lagrange variant.

    Returns:
        The Lagrange variant as a string.

    """
    return variant.name


def string_to_dpc_variant(variant: str) -> _DV:
    """Convert a string to a Basix DPCVariant enum.

    Args:
        variant: DPC variant as a string.

    Returns:
        The DPC variant.

    """
    if not hasattr(_DV, variant.lower()):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_DV, variant.lower())


def dpc_variant_to_string(variant: _DV) -> str:
    """Convert a Basix DPCVariant enum to a string.

    Args:
        variant: DPC variant.

    Returns:
        The DPC variant as a string.

    """
    return variant.name
