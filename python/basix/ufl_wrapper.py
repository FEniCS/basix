"""Functions to directly wrap Basix elements in UFL."""

from ufl.finiteelement.finiteelementbase import FiniteElementBase
import basix


class BasixElement(FiniteElementBase):
    """A wrapper allowing Basix elements to be used with UFL."""

    def __init__(self, element):
        super().__init__(
            element.family.name, element.cell_type.name, element.degree, None, tuple(element.value_shape),
            tuple(element.value_shape))

        if element.family == basix.ElementFamily.custom:
            signature = compute_hash(
                element.cell_type, element.degree, element.value_shape,
                element.wcoeffs, element.entity_transformations,
                element.x, element.M, element.map_type, element.discontinuous, element.degree_bounds)
            self._repr = f"custom Basix element ({signature})"
        else:
            self._repr = (f"Basix element ({element.family.name}, {element.cell_type.name}, {element.degree}, "
                          f"{element.lagrange_variant.name}, {element.dpc_variant.name}, {element.discontinuous})")
        self.basix_element = element

    def mapping(self):
        """Get the map type for this element."""
        return map_type_to_string(self.basix_element.map_type)


def map_type_to_string(map_type):
    """Convert map type to a UFL string."""
    if map_type == basix.MapType.identity:
        return "identity"
    if map_type == basix.MapType.contravariantPiola:
        return "contravariant Piola"
    if map_type == basix.MapType.covariantPiola:
        return "covariant Piola"
    if map_type == basix.MapType.doubleContravariantPiola:
        return "double contravariant Piola"
    if map_type == basix.MapType.doubleCovariantPiola:
        return "double covariant Piola"
    raise ValueError(f"Unsupported map type: {map_type}")


def compute_hash(*items):
    """Compute a hash of a list of items."""
    return hash("__".join([f"{i}" for i in items]))
