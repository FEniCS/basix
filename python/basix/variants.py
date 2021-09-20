"""Functions to manipulate variant types."""

from ._basixcpp import LagrangeVariant as _LV


def string_to_lagrange_variant(variant: str):
    """Convert a string to a Basix LagrangeVariant enum."""
    if not hasattr(_LV, variant):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(_LV, variant)


def lagrange_variant_to_string(variant: _LT):
    """Convert a Basix LagrangeVariant enum to a string."""
    return variant.name
