"""Functions to directly wrap Basix elements in UFL."""

from ufl.finiteelement.finiteelementbase import FiniteElementBase as _FiniteElementBase
import hashlib as _hashlib
import basix as _basix


class BasixElement(_FiniteElementBase):
    """A wrapper allowing Basix elements to be used with UFL."""

    def __init__(self, element: _basix.finite_element.FiniteElement):
        super().__init__(
            element.family.name, element.cell_type.name, element.degree, None, tuple(element.value_shape),
            tuple(element.value_shape))

        if element.family == _basix.ElementFamily.custom:
            self._repr = f"custom Basix element ({_compute_signature(element)})"
        else:
            self._repr = (f"Basix element ({element.family.name}, {element.cell_type.name}, {element.degree}, "
                          f"{element.lagrange_variant.name}, {element.dpc_variant.name}, {element.discontinuous})")
        self.basix_element = element

    def mapping(self) -> str:
        """Return the map type"""
        return _map_type_to_string(self.basix_element.map_type)

    def __eq__(self, other):
        """Check if two elements are equal."""
        if isinstance(other, BasixElement):
            return self.basix_element == other.basix_element
        return False


def _map_type_to_string(map_type: _basix.MapType) -> str:
    """Convert map type to a UFL string."""
    if map_type == _basix.MapType.identity:
        return "identity"
    if map_type == _basix.MapType.L2Piola:
        return "L2 Piola"
    if map_type == _basix.MapType.contravariantPiola:
        return "contravariant Piola"
    if map_type == _basix.MapType.covariantPiola:
        return "covariant Piola"
    if map_type == _basix.MapType.doubleContravariantPiola:
        return "double contravariant Piola"
    if map_type == _basix.MapType.doubleCovariantPiola:
        return "double covariant Piola"
    raise ValueError(f"Unsupported map type: {map_type}")


def _compute_signature(element: _basix.finite_element.FiniteElement):
    """Compute a signature of a custom element."""
    signature = (f"{element.cell_type.name}, {element.degree}, {element.value_shape}, {element.map_type.name}, "
                 f"{element.discontinuous}, {element.degree_bounds}, ")
    data = ",".join([f"{i}" for row in element.wcoeffs for i in row])
    data += "__"
    for entity in element.x:
        for points in entity:
            data = ",".join([f"{i}" for p in points for i in p])
            data += "_"
    data += "__"
    for entity in element.M:
        for matrices in entity:
            data = ",".join([f"{i}" for mat in matrices for row in mat for i in row])
            data += "_"
    data += "__"
    for mat in element.entity_transformations().values():
        data = ",".join([f"{i}" for row in mat for i in row])
        data += "__"
    signature += _hashlib.sha1(data.encode('utf-8')).hexdigest()
    return signature
