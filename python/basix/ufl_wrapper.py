"""Functions to directly wrap Basix elements in UFL."""

import ufl as _ufl
from ufl.finiteelement.finiteelementbase import FiniteElementBase as _FiniteElementBase
import hashlib as _hashlib
import basix as _basix
import typing as _typing


class BasixElement(_FiniteElementBase):
    """A wrapper allowing Basix elements to be used with UFL."""

    def __init__(self, element: _basix.finite_element.FiniteElement):
        """Create a basix element."""
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
        """Return the map type."""
        return _map_type_to_string(self.basix_element.map_type)

    def __eq__(self, other):
        """Check if two elements are equal."""
        if isinstance(other, BasixElement):
            return self.basix_element == other.basix_element
        return False

    def __hash__(self):
        """Return a hash."""
        return hash(self._repr)


def _map_type_to_string(map_type: _basix.MapType) -> str:
    """Convert map type to a UFL string.

    Args:
        map_type: A map type.

    Returns:
        A string representing this map type.
    """
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


def _compute_signature(element: _basix.finite_element.FiniteElement) -> str:
    """Compute a signature of a custom element.

    Args:
        element: A Basix custom element.

    Returns:
        A hash identifying this element.
    """
    assert element.family == _basix.ElementFamily.custom

    signature = (f"{element.cell_type.name}, {element.value_shape}, {element.map_type.name}, "
                 f"{element.discontinuous}, {element.highest_complete_degree}, {element.highest_degree}, ")
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


def create_element(family: _typing.Union[_basix.ElementFamily, str], cell: _typing.Union[_basix.CellType, str],
                   degree: int, lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
                   dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset, discontinuous=False) -> BasixElement:
    """Create a UFL element using Basix.

    Args:
        family: The element's family as a Basix enum or a string.
        cell: The cell type as a Basix enum or a string.
        degree: The degree of the finite element.
        lagrange_variant: The variant of Lagrange to be used.
        dpc_variant: The variant of DPC to be used.
        discontinuous: If set to True, the discontinuous version of this element will be created.
    """
    if isinstance(cell, str):
        cell = _basix.cell.string_to_type(cell)
    if isinstance(family, str):
        if family.startswith("Discontinuous "):
            family = family[14:]
            discontinuous = True
        if family in ["DP", "DG", "DQ"]:
            family = "P"
            discontinuous = True
        if family == "DPC":
            discontinuous = True

        family = _basix.finite_element.string_to_family(family, cell.name)

    e = _basix.create_element(family, cell, degree, lagrange_variant, dpc_variant, discontinuous)
    return BasixElement(e)


def create_vector_element(
    family: _typing.Union[_basix.ElementFamily, str], cell: _typing.Union[_basix.CellType, str],
    degree: int, lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
    dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset, discontinuous=False
) -> _ufl.VectorElement:
    """Create a UFL vector element using Basix.

    A vector element is an element which uses multiple copies of a scalar element to represent a
    vector-valued function.

    Args:
        family: The element's family as a Basix enum or a string.
        cell: The cell type as a Basix enum or a string.
        degree: The degree of the finite element.
        lagrange_variant: The variant of Lagrange to be used.
        dpc_variant: The variant of DPC to be used.
        discontinuous: If set to True, the discontinuous version of this element will be created.
    """
    e = create_element(family, cell, degree, lagrange_variant, dpc_variant, discontinuous)
    return _ufl.VectorElement(e)


def create_tensor_element(
    family: _typing.Union[_basix.ElementFamily, str], cell: _typing.Union[_basix.CellType, str],
    degree: int, lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
    dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset, discontinuous=False
) -> _ufl.TensorElement:
    """Create a UFL tensor element using Basix.

    A tensor element is an element which uses multiple copies of a scalar element to represent a
    tensor-valued function.

    Args:
        family: The element's family as a Basix enum or a string.
        cell: The cell type as a Basix enum or a string.
        degree: The degree of the finite element.
        lagrange_variant: The variant of Lagrange to be used.
        dpc_variant: The variant of DPC to be used.
        discontinuous: If set to True, the discontinuous version of this element will be created.
    """
    e = create_element(family, cell, degree, lagrange_variant, dpc_variant, discontinuous)
    return _ufl.TensorElement(e)
