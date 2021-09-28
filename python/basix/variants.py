"""Functions to manipulate variant types."""

from ._basixcpp import LagrangeVariant as _LV


def string_to_lagrange_variant(variant: str):
    """Convert a string to a Basix LagrangeVariant enum."""
    if variant == "gll":
        return _LV.gll_warped
    if variant == "chebyshev":
        return _LV.chebyshev_isaac
    if variant == "gl":
        return _LV.gl_isaac

    if not hasattr(_LV, variant):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_LV, variant)


def lagrange_variant_to_string(variant: _LV):
    """Convert a Basix LagrangeVariant enum to a string."""
    return variant.name
