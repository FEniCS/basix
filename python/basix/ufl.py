# Copyright (C) 2023-2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Functions to directly wrap Basix elements in UFL."""

import functools as _functools
import hashlib as _hashlib
import itertools as _itertools
import typing as _typing
from abc import abstractmethod as _abstractmethod
from abc import abstractproperty as _abstractproperty
from warnings import warn as _warn

import numpy as np
import numpy.typing as _npt
import ufl as _ufl
from ufl.finiteelement import AbstractFiniteElement as _AbstractFiniteElement
from ufl.pullback import AbstractPullback as _AbstractPullback
from ufl.pullback import IdentityPullback as _IdentityPullback
from ufl.pullback import MixedPullback as _MixedPullback
from ufl.pullback import SymmetricPullback as _SymmetricPullback
from ufl.pullback import UndefinedPullback as _UndefinedPullback

import basix as _basix

__all__ = [
    "element",
    "enriched_element",
    "custom_element",
    "mixed_element",
    "quadrature_element",
    "real_element",
    "blocked_element",
    "wrap_element",
]

_spacemap = {
    _basix.SobolevSpace.L2: _ufl.sobolevspace.L2,
    _basix.SobolevSpace.H1: _ufl.sobolevspace.H1,
    _basix.SobolevSpace.H2: _ufl.sobolevspace.H2,
    _basix.SobolevSpace.HInf: _ufl.sobolevspace.HInf,
    _basix.SobolevSpace.HDiv: _ufl.sobolevspace.HDiv,
    _basix.SobolevSpace.HCurl: _ufl.sobolevspace.HCurl,
    _basix.SobolevSpace.HEin: _ufl.sobolevspace.HEin,
    _basix.SobolevSpace.HDivDiv: _ufl.sobolevspace.HDivDiv,
}

_pullbackmap = {
    _basix.MapType.identity: _ufl.identity_pullback,
    _basix.MapType.L2Piola: _ufl.l2_piola,
    _basix.MapType.contravariantPiola: _ufl.contravariant_piola,
    _basix.MapType.covariantPiola: _ufl.covariant_piola,
    _basix.MapType.doubleContravariantPiola: _ufl.double_contravariant_piola,
    _basix.MapType.doubleCovariantPiola: _ufl.double_covariant_piola,
}


def _ufl_sobolev_space_from_enum(s: _basix.SobolevSpace):
    """Convert a Basix Sobolev space enum to a UFL Sobolev space.

    Args:
        s: The Basix Sobolev space

    Returns:
        UFL Sobolev space
    """
    if s not in _spacemap:
        raise ValueError(f"Could not convert to UFL Sobolev space: {s.name}")
    return _spacemap[s]


def _ufl_pullback_from_enum(m: _basix.maps.MapType) -> _AbstractPullback:
    """Convert an enum to a UFL pull back.

    Args:
        m: A map type.

    Returns:
        UFL pull back.

    """
    if m not in _pullbackmap:
        raise ValueError(f"Could not convert to UFL pull back: {m.name}")
    return _pullbackmap[m]


class _ElementBase(_AbstractFiniteElement):
    """A base wrapper to allow elements to be used with UFL.

    This class includes methods and properties needed by UFL and FFCx.
    This is a base class containing functions common to all the element
    types defined in this file.
    """

    def __init__(
        self,
        repr: str,
        cellname: str,
        reference_value_shape: tuple[int, ...],
        degree: int = -1,
        pullback: _AbstractPullback = _UndefinedPullback(),
    ):
        """Initialise the element."""
        self._repr = repr
        self._cellname = cellname
        self._reference_value_shape = reference_value_shape
        self._degree = degree
        self._pullback = pullback

    # Implementation of methods for UFL AbstractFiniteElement
    def __repr__(self):
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self):
        """Format as string for nice printing."""
        return self._repr

    def __hash__(self) -> int:
        """Return a hash."""
        return hash("basix" + self._repr)

    @_abstractmethod
    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""

    @property
    def sobolev_space(self):
        """Underlying Sobolev space."""
        return _ufl_sobolev_space_from_enum(self.basix_sobolev_space)

    @property
    def pullback(self) -> _AbstractPullback:
        """Pullback for this element."""
        return self._pullback

    @_abstractproperty
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return None.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """

    @_abstractproperty
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """

    @property
    def cell(self) -> _ufl.Cell:
        """Cell of the finite element."""
        return _ufl.cell.Cell(self._cellname)

    @property
    def reference_value_shape(self) -> tuple[int, ...]:
        """Shape of the value space on the reference cell."""
        return self._reference_value_shape

    @property
    def sub_elements(self) -> list[_AbstractFiniteElement]:
        """List of sub elements.

        This function does not recurse: i.e. it does not extract the
        sub-elements of sub-elements.
        """
        return []

    # Basix specific functions
    @_abstractmethod
    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions
        """

    @_abstractmethod
    def get_component_element(self, flat_component: int) -> tuple[_typing.Any, int, int]:
        """Get element that represents a component, and the offset and stride of the component.

        For example, for a mixed element, this will return the
        sub-element that represents the given component, the offset of
        that sub-element, and a stride of 1. For a blocked element, this
        will return the sub-element, an offset equal to the component
        number, and a stride equal to the block size. For vector-valued
        element (eg H(curl) and H(div) elements), this returns a
        component element (and as offset of 0 and a stride of 1). When
        tabulate is called on the component element, only the part of the
        table for the given component is returned.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """

    @_abstractproperty
    def ufcx_element_type(self) -> str:
        """Element type."""

    @_abstractproperty
    def dim(self) -> int:
        """Number of DOFs the element has."""

    @_abstractproperty
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""

    @_abstractproperty
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""

    @_abstractproperty
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""

    @_abstractproperty
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""

    @_abstractproperty
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""

    @_abstractproperty
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""

    @_abstractproperty
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""

    @_abstractproperty
    def family_name(self) -> str:
        """Family name of the element."""

    @_abstractproperty
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""

    @_abstractproperty
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""

    @_abstractproperty
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""

    @_abstractproperty
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""

    @_abstractproperty
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""

    @_abstractproperty
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""

    @_abstractproperty
    def polyset_type(self) -> _basix.PolysetType:
        """The polyset type of the element."""

    @_abstractproperty
    def basix_sobolev_space(self):
        """Return a Basix enum representing the underlying Sobolev space."""

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        return None

    @property
    def degree(self) -> int:
        """The degree of the element."""
        return self._degree

    def custom_quadrature(self) -> tuple[_npt.NDArray[np.float64], _npt.NDArray[np.float64]]:
        """Return custom quadrature rule or raise a ValueError."""
        raise ValueError("Element does not have a custom quadrature rule.")

    @property
    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be
        computed as a tensor product of the basis elements of the
        elements in the factoriaation.
        """
        return False

    @property
    def block_size(self) -> int:
        """The block size of the element."""
        return 1

    @property
    def _wcoeffs(self) -> _npt.NDArray[np.float64]:
        """The coefficients used to define the polynomial set."""
        raise NotImplementedError()

    @property
    def _x(self) -> list[list[_npt.NDArray[np.float64]]]:
        """The points used to define interpolation."""
        raise NotImplementedError()

    @property
    def _M(self) -> list[list[_npt.NDArray[np.float64]]]:
        """The matrices used to define interpolation."""
        raise NotImplementedError()

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        raise NotImplementedError()

    @property
    def is_custom_element(self) -> bool:
        """True if the element is a custom Basix element."""
        return False

    @property
    def has_custom_quadrature(self) -> bool:
        """True if the element has a custom quadrature rule."""
        return False

    @property
    def basix_element(self):
        """Underlying Basix element."""
        raise NotImplementedError()


class _BasixElement(_ElementBase):
    """A wrapper allowing Basix elements to be used directly with UFL.

    This class allows elements created with `basix.create_element` to be
    wrapped as UFL compatible elements. Users should not directly call
    this class's initiliser, but should use the `element` function
    instead.
    """

    _element: _basix.finite_element.FiniteElement

    def __init__(self, element: _basix.finite_element.FiniteElement):
        """Create a Basix element."""
        if element.family == _basix.ElementFamily.custom:
            self._is_custom = True
            repr = f"custom Basix element ({_compute_signature(element)})"
        else:
            self._is_custom = False
            repr = (
                f"Basix element ({element.family.name}, {element.cell_type.name}, "
                f"{element.degree}, "
                f"{element.lagrange_variant.name}, {element.dpc_variant.name}, "
                f"{element.discontinuous}, "
                f"{element.dtype}, {element.dof_ordering})"
            )

        super().__init__(
            repr,
            element.cell_type.name,
            tuple(element.value_shape),
            element.degree,
            _ufl_pullback_from_enum(element.map_type),
        )

        self._element = element

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, _BasixElement) and self._element == other._element

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        tab = self._element.tabulate(nderivs, points)
        # TODO: update FFCx to remove the need for transposing here
        return tab.transpose((0, 1, 3, 2)).reshape((tab.shape[0], tab.shape[1], -1))

    def get_component_element(self, flat_component: int) -> tuple[_ElementBase, int, int]:
        """Get element that represents a component.

        Element that represents a component of the element, and the
        offset and stride of the component.

        For example, for a mixed element, this will return the
        sub-element that represents the given component, the offset of
        that sub-element, and a stride of 1. For a blocked element, this
        will return the sub-element, an offset equal to the component
        number, and a stride equal to the block size. For vector-valued
        element (eg H(curl) and H(div) elements), this returns a
        component element (and as offset of 0 and a stride of 1). When
        tabulate is called on the component element, only the part of
        the table for the given component is returned.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        assert flat_component < self.reference_value_size
        return _ComponentElement(self, flat_component), 0, 1

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        if not self.has_tensor_product_factorisation:
            return None
        return self._element.get_tensor_product_representation()

    @property
    def basix_sobolev_space(self):
        """Return a Basix enum representing the underlying Sobolev space."""
        return self._element.sobolev_space

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        if self._is_custom:
            return "ufcx_basix_custom_element"
        else:
            return "ufcx_basix_element"

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        return self._element.dim

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""
        return self._element.num_entity_dofs

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""
        return self._element.entity_dofs

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return self._element.num_entity_closure_dofs

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""
        return self._element.entity_closure_dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return 0

    @property
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""
        return _basix.topology(self._element.cell_type)

    @property
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""
        return _basix.geometry(self._element.cell_type)

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return self._element.family.name

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return self._element.family

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return self._element.lagrange_variant

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return self._element.dpc_variant

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self._element.cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self._element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self._element.interpolation_nderivs

    @property
    def is_custom_element(self) -> bool:
        """True if the element is a custom Basix element."""
        return self._is_custom

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        return self._element.map_type

    @property
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return None.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._element.embedded_superdegree

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._element.embedded_subdegree

    @property
    def polyset_type(self) -> _basix.PolysetType:
        return self._element.polyset_type

    @property
    def _wcoeffs(self) -> _npt.NDArray[np.float64]:
        """The coefficients used to define the polynomial set."""
        return self._element.wcoeffs

    @property
    def _x(self) -> list[list[_npt.NDArray[np.float64]]]:
        """The points used to define interpolation."""
        return self._element.x

    @property
    def _M(self) -> list[list[_npt.NDArray[np.float64]]]:
        """The matrices used to define interpolation."""
        return self._element.M

    @property
    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be
        computed as a tensor product of the basis elements of the
        elements in the factorisation.

        """
        return self._element.has_tensor_product_factorisation

    @property
    def basix_element(self):
        """Underlying Basix element."""
        return self._element


class _ComponentElement(_ElementBase):
    """An element representing one component of a _BasixElement.

    This element type is used when UFL's ``get_component_element``
    function is called.

    """

    _element: _ElementBase
    _component: int

    def __init__(self, element: _ElementBase, component: int):
        """Initialise the element."""
        self._element = element
        self._component = component
        repr = f"component element ({element!r}, {component}"
        repr += ")"
        super().__init__(repr, element.cell_type.name, (1,), element._degree)

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (
            isinstance(other, _ComponentElement)
            and self._element == other._element
            and self._component == other._component
        )

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at.

        Returns:
            Tabulated basis functions.
        """
        tables = self._element.tabulate(nderivs, points)
        output = []
        for tbl in tables:
            shape = (points.shape[0], *self._element._reference_value_shape, -1)
            tbl = tbl.reshape(shape)
            if len(self._element._reference_value_shape) == 0:
                output.append(tbl)
            elif len(self._element._reference_value_shape) == 1:
                output.append(tbl[:, self._component, :])
            elif len(self._element._reference_value_shape) == 2:
                if isinstance(self._element, _BlockedElement) and self._element._has_symmetry:
                    # FIXME: check that this behaves as expected
                    output.append(tbl[:, self._component, :])
                else:
                    vs0 = self._element._reference_value_shape[0]
                    output.append(tbl[:, self._component // vs0, self._component % vs0, :])
            else:
                raise NotImplementedError()
        return np.asarray(output, dtype=np.float64)

    def get_component_element(self, flat_component: int) -> tuple[_ElementBase, int, int]:
        """Get element that represents a component.

        Element that represents a component of the element, and the
        offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """
        if flat_component == 0:
            return self, 0, 1
        else:
            raise NotImplementedError()

    @property
    def basix_sobolev_space(self):
        """Return a Basix enum representing the underlying Sobolev space."""
        return self._element.basix_sobolev_space

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        raise NotImplementedError()

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""
        raise NotImplementedError()

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""
        raise NotImplementedError()

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""
        raise NotImplementedError()

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""
        raise NotImplementedError()

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        raise NotImplementedError()

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        raise NotImplementedError()

    @property
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""
        raise NotImplementedError()

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return self._element.element_family

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return self._element.lagrange_variant

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return self._element.dpc_variant

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self._element.cell_type

    @property
    def polyset_type(self) -> _basix.PolysetType:
        return self._element.polyset_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self._element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self._element.interpolation_nderivs

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        raise NotImplementedError()

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        raise NotImplementedError()

    @property
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return ``None``.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._element.embedded_superdegree

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._element.embedded_subdegree

    @property
    def basix_element(self):
        """Underlying Basix element."""
        return self._element


class _MixedElement(_ElementBase):
    """A mixed element that combines two or more elements.

    This can be used when multiple different elements appear in a form.
    Users should not directly call this class's initilizer, but should
    use the :func:`mixed_element` function instead.
    """

    _sub_elements: list[_ElementBase]

    def __init__(self, sub_elements: list[_ElementBase]):
        """Initialise the element."""
        assert len(sub_elements) > 0
        self._sub_elements = sub_elements
        if all(isinstance(e.pullback, _IdentityPullback) for e in sub_elements):
            pullback = _ufl.identity_pullback
        else:
            pullback = _MixedPullback(self)

        repr = "mixed element (" + ", ".join(i._repr for i in sub_elements) + ")"
        super().__init__(
            repr,
            sub_elements[0].cell_type.name,
            (sum(i.reference_value_size for i in sub_elements),),
            pullback=pullback,
        )

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        if isinstance(other, _MixedElement) and len(self._sub_elements) == len(other._sub_elements):
            for i, j in zip(self._sub_elements, other._sub_elements):
                if i != j:
                    return False
            return True
        return False

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    @property
    def degree(self) -> int:
        """Degree of the element."""
        return max((e.degree for e in self._sub_elements), default=-1)

    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions
        """
        tables = []
        results = [e.tabulate(nderivs, points) for e in self._sub_elements]
        for deriv_tables in zip(*results):
            new_table = np.zeros((len(points), self.reference_value_size * self.dim))
            start = 0
            for e, t in zip(self._sub_elements, deriv_tables):
                for i in range(0, e.dim, e.reference_value_size):
                    new_table[:, start : start + e.reference_value_size] = t[
                        :, i : i + e.reference_value_size
                    ]
                    start += self.reference_value_size
            tables.append(new_table)
        return np.asarray(tables, dtype=np.float64)

    def get_component_element(self, flat_component: int) -> tuple[_ElementBase, int, int]:
        """Get element that represents a component.

        Element that represents a component of the element, and the
        offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """
        sub_dims = [0] + [e.dim for e in self._sub_elements]
        sub_cmps = [0] + [e.reference_value_size for e in self._sub_elements]

        irange = np.cumsum(sub_dims)
        crange = np.cumsum(sub_cmps)

        # Find index of sub element which corresponds to the current
        # flat component
        component_element_index = np.where(crange <= flat_component)[0].shape[0] - 1

        sub_e = self._sub_elements[component_element_index]

        e, offset, stride = sub_e.get_component_element(
            flat_component - crange[component_element_index]
        )
        # TODO: is this offset correct?
        return e, irange[component_element_index] + offset, stride

    @property
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return None.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return max(e.embedded_superdegree for e in self._sub_elements)

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        raise NotImplementedError()

    @property
    def map_type(self) -> _basix.MapType:
        """Basix map type."""
        raise NotImplementedError()

    @property
    def basix_sobolev_space(self):
        """Basix Sobolev space that the element belongs to."""
        return _basix.sobolev_spaces.intersection(
            [e.basix_sobolev_space for e in self._sub_elements]
        )

    @property
    def sub_elements(self) -> list[_ElementBase]:
        """List of sub elements."""
        return self._sub_elements

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return "ufcx_mixed_element"

    @property
    def dim(self) -> int:
        """Dimension (number of DOFs) for the element."""
        return sum(e.dim for e in self._sub_elements)

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""
        data = [e.num_entity_dofs for e in self._sub_elements]
        return [
            [sum(d[tdim][entity_n] for d in data) for entity_n, _ in enumerate(entities)]
            for tdim, entities in enumerate(data[0])
        ]

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""
        dofs: list[list[list[int]]] = [
            [[] for i in entities] for entities in self._sub_elements[0].entity_dofs
        ]
        start_dof = 0
        for e in self._sub_elements:
            for tdim, entities in enumerate(e.entity_dofs):
                for entity_n, entity_dofs in enumerate(entities):
                    dofs[tdim][entity_n] += [start_dof + i for i in entity_dofs]
            start_dof += e.dim
        return dofs

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""
        data = [e.num_entity_closure_dofs for e in self._sub_elements]
        return [
            [sum(d[tdim][entity_n] for d in data) for entity_n, _ in enumerate(entities)]
            for tdim, entities in enumerate(data[0])
        ]

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""
        dofs: list[list[list[int]]] = [
            [[] for i in entities] for entities in self._sub_elements[0].entity_closure_dofs
        ]
        start_dof = 0
        for e in self._sub_elements:
            for tdim, entities in enumerate(e.entity_closure_dofs):
                for entity_n, entity_dofs in enumerate(entities):
                    dofs[tdim][entity_n] += [start_dof + i for i in entity_dofs]
            start_dof += e.dim
        return dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Number of global support DOFs."""
        return sum(e.num_global_support_dofs for e in self._sub_elements)

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return "mixed element"

    @property
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""
        return self._sub_elements[0].reference_topology

    @property
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""
        return self._sub_elements[0].reference_geometry

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return None

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return None

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return None

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self._sub_elements[0].cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return False

    @property
    def interpolation_nderivs(self) -> int:
        """Number of derivatives needed when interpolating."""
        return max([e.interpolation_nderivs for e in self._sub_elements])

    @property
    def polyset_type(self) -> _basix.PolysetType:
        pt = _basix.PolysetType.standard
        for e in self._sub_elements:
            pt = _basix.polyset_superset(self.cell_type, pt, e.polyset_type)
        return pt

    def custom_quadrature(self) -> tuple[_npt.NDArray[np.float64], _npt.NDArray[np.float64]]:
        """Return custom quadrature rule or raise a ValueError."""
        custom_q = None
        for e in self._sub_elements:
            if e.has_custom_quadrature:
                if custom_q is None:
                    custom_q = e.custom_quadrature()
                else:
                    p, w = e.custom_quadrature()
                    if not np.allclose(p, custom_q[0]) or not np.allclose(w, custom_q[1]):
                        raise ValueError(
                            "Subelements of mixed element use different quadrature rules"
                        )
        if custom_q is not None:
            return custom_q
        raise ValueError("Element does not have custom quadrature")

    @property
    def has_custom_quadrature(self) -> bool:
        """True if the element has a custom quadrature rule."""
        for e in self._sub_elements:
            if e.has_custom_quadrature:
                return True
        return False


class _BlockedElement(_ElementBase):
    """Element with a block size that contains multiple copies of a sub element.

    This can be used to (for example) create vector and tensor Lagrange
    elements. Users should not directly call this classes initilizer,
    but should use the `blocked_element` function instead.

    """

    _block_shape: tuple[int, ...]
    _sub_element: _ElementBase
    _block_size: int

    def __init__(
        self,
        sub_element: _ElementBase,
        shape: tuple[int, ...],
        symmetry: _typing.Optional[bool] = None,
    ):
        """Initialise the element."""
        if sub_element.reference_value_size != 1:
            raise ValueError(
                "Blocked elements of non-scalar elements are not supported. "
                "Try using _MixedElement instead."
            )
        if symmetry is not None:
            if len(shape) != 2:
                raise ValueError("symmetry argument can only be passed to elements of rank 2.")
            if shape[0] != shape[1]:
                raise ValueError("symmetry argument can only be passed to square shaped elements.")

        if symmetry:
            block_size = shape[0] * (shape[0] + 1) // 2
            self._has_symmetry = True
        else:
            block_size = 1
            for i in shape:
                block_size *= i
            self._has_symmetry = False
        assert block_size > 0

        self._sub_element = sub_element
        self._block_size = block_size
        self._block_shape = shape

        repr = f"blocked element ({sub_element!r}, {shape}"
        if symmetry is not None:
            repr += f", symmetry={symmetry}"
        repr += ")"

        super().__init__(
            repr,
            sub_element.cell_type.name,
            shape,
            sub_element._degree,
            sub_element._pullback,
        )

        if symmetry:
            n = 0
            symmetry_mapping = {}
            for i in range(shape[0]):
                for j in range(i + 1):
                    symmetry_mapping[(i, j)] = n
                    symmetry_mapping[(j, i)] = n
                    n += 1

            self._pullback = _SymmetricPullback(self, symmetry_mapping)

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (
            isinstance(other, _BlockedElement)
            and self._block_size == other._block_size
            and self._block_shape == other._block_shape
            and self._sub_element == other._sub_element
        )

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        assert len(self._block_shape) == 1  # TODO: block shape
        assert self.reference_value_size == self._block_size  # TODO: remove this assumption
        output = []
        for table in self._sub_element.tabulate(nderivs, points):
            # Repeat sub element horizontally
            assert len(table.shape) == 2
            new_table = np.zeros(
                (table.shape[0], *self._block_shape, self._block_size * table.shape[1])
            )
            for i, j in enumerate(_itertools.product(*[range(s) for s in self._block_shape])):
                if len(j) == 1:
                    new_table[:, j[0], i :: self._block_size] = table
                elif len(j) == 2:
                    new_table[:, j[0], j[1], i :: self._block_size] = table
                else:
                    raise NotImplementedError()
            output.append(new_table)
        return np.asarray(output, dtype=np.float64)

    def get_component_element(self, flat_component: int) -> tuple[_ElementBase, int, int]:
        """Get element that represents a component.

        Element that represents a component of the element, and the
        offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        return self._sub_element, flat_component, self._block_size

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        if not self.has_tensor_product_factorisation:
            return None
        return self._sub_element.get_tensor_product_representation()

    @property
    def block_size(self) -> int:
        """Block size of the element."""
        return self._block_size

    @property
    def reference_value_shape(self) -> tuple[int, ...]:
        """Reference value shape of the element basis function."""
        if self._has_symmetry:
            assert len(self._block_shape) == 2 and self._block_shape[0] == self._block_shape[1]
            return (self._block_shape[0] * (self._block_shape[0] + 1) // 2,)
        return self._reference_value_shape

    @property
    def basix_sobolev_space(self):
        """Basix enum representing the underlying Sobolev space."""
        return self._sub_element.basix_sobolev_space

    @property
    def sub_elements(self) -> list[_ElementBase]:
        """List of sub elements."""
        return [self._sub_element for _ in range(self._block_size)]

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return self._sub_element.ufcx_element_type

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        return self._sub_element.dim * self._block_size

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""
        return [[j * self._block_size for j in i] for i in self._sub_element.num_entity_dofs]

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""
        # TODO: should this return this, or should it take blocks into
        # account?
        return [
            [[k * self._block_size + b for k in j for b in range(self._block_size)] for j in i]
            for i in self._sub_element.entity_dofs
        ]

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return [
            [j * self._block_size for j in i] for i in self._sub_element.num_entity_closure_dofs
        ]

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""
        # TODO: should this return this, or should it take blocks into
        # account?
        return [
            [[k * self._block_size + b for k in j for b in range(self._block_size)] for j in i]
            for i in self._sub_element.entity_closure_dofs
        ]

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return self._sub_element.num_global_support_dofs * self._block_size

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return self._sub_element.family_name

    @property
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""
        return self._sub_element.reference_topology

    @property
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""
        return self._sub_element.reference_geometry

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return self._sub_element.lagrange_variant

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return self._sub_element.dpc_variant

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return self._sub_element.element_family

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self._sub_element.cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self._sub_element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self._sub_element.interpolation_nderivs

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        return self._sub_element.map_type

    @property
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return None.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._sub_element.embedded_superdegree

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._sub_element.embedded_subdegree

    @property
    def polyset_type(self) -> _basix.PolysetType:
        return self._sub_element.polyset_type

    @property
    def _wcoeffs(self) -> _npt.NDArray[np.float64]:
        """Coefficients used to define the polynomial set."""
        sub_wc = self._sub_element._wcoeffs
        wcoeffs = np.zeros((sub_wc.shape[0] * self._block_size, sub_wc.shape[1] * self._block_size))
        for i in range(self._block_size):
            wcoeffs[
                sub_wc.shape[0] * i : sub_wc.shape[0] * (i + 1),
                sub_wc.shape[1] * i : sub_wc.shape[1] * (i + 1),
            ] = sub_wc
        return wcoeffs

    @property
    def _x(self) -> list[list[_npt.NDArray[np.float64]]]:
        """Points used to define interpolation."""
        return self._sub_element._x

    @property
    def _M(self) -> list[list[_npt.NDArray[np.float64]]]:
        """Matrices used to define interpolation."""
        M = []
        for M_list in self._sub_element._M:
            M_row = []
            for mat in M_list:
                new_mat = np.zeros(
                    (
                        mat.shape[0] * self._block_size,
                        mat.shape[1] * self._block_size,
                        mat.shape[2],
                        mat.shape[3],
                    )
                )
                for i in range(self._block_size):
                    new_mat[
                        i * mat.shape[0] : (i + 1) * mat.shape[0],
                        i * mat.shape[1] : (i + 1) * mat.shape[1],
                        :,
                        :,
                    ] = mat
                M_row.append(new_mat)
            M.append(M_row)
        return M

    @property
    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be
        computed as a tensor product of the basis elements of the
        elements in the factoriaation.
        """
        return self._sub_element.has_tensor_product_factorisation

    def custom_quadrature(self) -> tuple[_npt.NDArray[np.float64], _npt.NDArray[np.float64]]:
        """Return custom quadrature rule or raise a ValueError."""
        return self._sub_element.custom_quadrature()

    @property
    def has_custom_quadrature(self) -> bool:
        """True if the element has a custom quadrature rule."""
        return self._sub_element.has_custom_quadrature

    @property
    def basix_element(self):
        """Underlying Basix element."""
        return self._sub_element.basix_element


class _QuadratureElement(_ElementBase):
    """A quadrature element."""

    def __init__(
        self,
        cell: _basix.CellType,
        points: _npt.NDArray[np.float64],
        weights: _npt.NDArray[np.float64],
        pullback: _AbstractPullback,
        degree: _typing.Optional[int] = None,
    ):
        """Initialise the element."""
        self._points = points
        self._weights = weights
        repr = f"QuadratureElement({cell.name}, {points!r}, {weights!r}, {pullback})".replace(
            "\n", ""
        )
        self._cell_type = cell
        self._entity_counts = [len(i) for i in _basix.topology(cell)]

        if degree is None:
            degree = len(points)

        super().__init__(repr, cell.name, (), degree, pullback=pullback)

    def basix_sobolev_space(self):
        """Underlying Sobolev space."""
        return _basix.sobolev_spaces.L2

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, _QuadratureElement) and (
            self._cell_type == other._cell_type
            and self._pullback == other._pullback
            and self._points.shape == other._points.shape
            and self._weights.shape == other._weights.shape
            and np.allclose(self._points, other._points)
            and np.allclose(self._weights, other._weights)
        )

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions
        """
        if nderivs > 0:
            raise ValueError("Cannot take derivatives of Quadrature element.")

        if points.shape != self._points.shape:
            raise ValueError("Mismatch of tabulation points and element points.")
        tables = np.asarray([np.eye(points.shape[0], points.shape[0])])
        return tables

    def get_component_element(self, flat_component: int) -> tuple[_ElementBase, int, int]:
        """Get element that represents a component.

        Element that represents a component of the element, and the
        offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """
        return self, 0, 1

    def custom_quadrature(self) -> tuple[_npt.NDArray[np.float64], _npt.NDArray[np.float64]]:
        """Return custom quadrature rule or raise a ValueError."""
        return self._points, self._weights

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return "ufcx_quadrature_element"

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        return self._points.shape[0]

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""
        dofs = []
        for d in self._entity_counts[:-1]:
            dofs += [[0] * d]

        dofs += [[self.dim]]
        return dofs

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""
        start_dof = 0
        entity_dofs = []
        for i in self.num_entity_dofs:
            dofs_list = []
            for j in i:
                dofs_list.append([start_dof + k for k in range(j)])
                start_dof += j
            entity_dofs.append(dofs_list)
        return entity_dofs

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return self.num_entity_dofs

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""
        return self.entity_dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return 0

    @property
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""
        raise NotImplementedError()

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return "quadrature"

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return None

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return None

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return None

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self._cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return False

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        return _basix.MapType.identity

    @property
    def polyset_type(self) -> _basix.PolysetType:
        """The polyset type of the element."""
        raise NotImplementedError()

    @property
    def has_custom_quadrature(self) -> bool:
        """True if the element has a custom quadrature rule."""
        return True

    @property
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return ``None``.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self.degree

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return -1


class _RealElement(_ElementBase):
    """A real element."""

    def __init__(self, cell: _basix.CellType, value_shape: tuple[int, ...]):
        """Initialise the element."""
        self._cell_type = cell
        tdim = len(_basix.topology(cell)) - 1

        super().__init__(f"RealElement({cell.name}, {value_shape})", cell.name, value_shape, 0)

        self._entity_counts = []
        if tdim >= 1:
            self._entity_counts.append(self.cell.num_vertices())
        if tdim >= 2:
            self._entity_counts.append(self.cell.num_edges())
        if tdim >= 3:
            self._entity_counts.append(self.cell.num_facets())
        self._entity_counts.append(1)

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (
            isinstance(other, _RealElement)
            and self._cell_type == other._cell_type
            and self._reference_value_shape == other._reference_value_shape
        )

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[np.float64]) -> _npt.NDArray[np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        out = np.zeros((nderivs + 1, len(points), self.reference_value_size**2))
        for v in range(self.reference_value_size):
            out[0, :, self.reference_value_size * v + v] = 1.0
        return out

    def get_component_element(self, flat_component: int) -> tuple[_ElementBase, int, int]:
        """Get element that represents a component.

        Element that represents a component of the element, and the
        offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        assert flat_component < self.reference_value_size
        return self, 0, 1

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return "ufcx_real_element"

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        return 0

    @property
    def embedded_superdegree(self) -> int:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return ``None``.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return 0

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return 0

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with each entity."""
        dofs = []
        for d in self._entity_counts[:-1]:
            dofs += [[0] * d]

        dofs += [[self.dim]]
        return dofs

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with each entity."""
        start_dof = 0
        entity_dofs = []
        for i in self.num_entity_dofs:
            dofs_list = []
            for j in i:
                dofs_list.append([start_dof + k for k in range(j)])
                start_dof += j
            entity_dofs.append(dofs_list)
        return entity_dofs

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return self.num_entity_dofs

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """DOF numbers associated with the closure of each entity."""
        return self.entity_dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return 1

    @property
    def reference_topology(self) -> list[list[list[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _npt.NDArray[np.float64]:
        """Geometry of the reference element."""
        raise NotImplementedError()

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return "real"

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return None

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return None

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return None

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self._cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return False

    @property
    def basix_sobolev_space(self):
        """Underlying Sobolev space."""
        return _basix.sobolev_spaces.Hinf

    @property
    def map_type(self) -> _basix.MapType:
        """Basix map type."""
        return _basix.MapType.identity

    @property
    def polyset_type(self) -> _basix.PolysetType:
        """Polyset type of the element."""
        raise NotImplementedError()


def _compute_signature(element: _basix.finite_element.FiniteElement) -> str:
    """Compute a signature of a custom element.

    Args:
        element: A Basix custom element.

    Returns:
        A hash identifying this element.
    """
    assert element.family == _basix.ElementFamily.custom
    signature = (
        f"{element.cell_type.name}, {element.value_shape}, {element.map_type.name}, "
        f"{element.discontinuous}, {element.embedded_subdegree}, {element.embedded_superdegree}, "
        f"{element.dtype}, {element.dof_ordering}"
    )
    data = ",".join([f"{i}" for row in element.wcoeffs for i in row])
    data += "__"
    for entity in element.x:
        for points in entity:
            data += ",".join([f"{i}" for p in points for i in p])
            data += "_"
    data += "__"

    for entity in element.M:
        for matrices in entity:
            data += ",".join([f"{i}" for mat in matrices for row in mat for i in row])
            data += "_"
    data += "__"

    for mat in element.entity_transformations().values():
        data += ",".join([f"{i}" for row in mat for i in row])
        data += "__"
    signature += _hashlib.sha1(data.encode("utf-8")).hexdigest()

    return signature


@_functools.cache
def element(
    family: _typing.Union[_basix.ElementFamily, str],
    cell: _typing.Union[_basix.CellType, str],
    degree: int,
    lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
    dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset,
    discontinuous: bool = False,
    shape: _typing.Optional[tuple[int, ...]] = None,
    symmetry: _typing.Optional[bool] = None,
    dtype: _npt.DTypeLike = np.float64,
) -> _ElementBase:
    """Create a UFL compatible element using Basix.

    Args:
        family: Element family/type.
        cell: Element cell type.
        degree: Degree of the finite element.
        lagrange_variant: Variant of Lagrange to be used.
        dpc_variant: Variant of DPC to be used.
        discontinuous: If ``True``, the discontinuous version of the
            element is created.
        shape: Value shape of the element. For scalar-valued families,
            this can be used to create vector and tensor elements.
        symmetry: Set to ``True`` if the tensor is symmetric. Valid for
            rank 2 elements only.
        dtype: Floating point data type.

    Returns:
        A finite element.

    """
    # Conversion of string arguments to types
    if isinstance(cell, str):
        cell = _basix.cell.string_to_type(cell)
    if isinstance(family, str):
        if family.startswith("Discontinuous "):
            family = family[14:]
            discontinuous = True
        if family in ["DP", "DG", "DQ"]:
            family = "P"
            discontinuous = True
        if family == "CG":
            _warn(
                '"CG" element name is deprecated. Consider using "Lagrange" or "P" instead',
                DeprecationWarning,
                stacklevel=2,
            )
            family = "P"
            discontinuous = False
        if family == "DPC":
            discontinuous = True

        family = _basix.finite_element.string_to_family(family, cell.name)

    # Default variant choices
    EF = _basix.ElementFamily
    if lagrange_variant == _basix.LagrangeVariant.unset:
        if family == EF.P:
            lagrange_variant = _basix.LagrangeVariant.gll_warped
        elif family in [EF.RT, EF.N1E]:
            lagrange_variant = _basix.LagrangeVariant.legendre
        elif family in [EF.serendipity, EF.BDM, EF.N2E]:
            lagrange_variant = _basix.LagrangeVariant.legendre

    if dpc_variant == _basix.DPCVariant.unset:
        if family in [EF.serendipity, EF.BDM, EF.N2E]:
            dpc_variant = _basix.DPCVariant.legendre
        elif family == EF.DPC:
            dpc_variant = _basix.DPCVariant.diagonal_gll

    e = _basix.create_element(
        family, cell, degree, lagrange_variant, dpc_variant, discontinuous, dtype=dtype
    )
    ufl_e = _BasixElement(e)

    if shape is None or shape == tuple(e.value_shape):
        if symmetry is not None:
            raise ValueError("Cannot pass a symmetry argument to this element.")
        return ufl_e
    else:
        return blocked_element(ufl_e, shape=shape, symmetry=symmetry)


def enriched_element(
    elements: list[_ElementBase],
    map_type: _typing.Optional[_basix.MapType] = None,
) -> _ElementBase:
    """Create an UFL compatible enriched element from a list of elements.

    Args:
        elements: The list of elements
        map_type: The map type for the enriched element.

    Returns:
        An enriched finite element.

    """
    ct = elements[0].cell_type
    ptype = elements[0].polyset_type
    vshape = elements[0].reference_value_shape
    vsize = elements[0].reference_value_size
    if map_type is None:
        map_type = elements[0].map_type
        for e in elements:
            if e.map_type != map_type:
                raise ValueError("Enriched elements on different map types not supported.")

    hcd = min(e.embedded_subdegree for e in elements)
    hd = max(e.embedded_superdegree for e in elements)
    ss = _basix.sobolev_spaces.intersection([e.basix_sobolev_space for e in elements])
    discontinuous = True
    for e in elements:
        if not e.discontinuous:
            discontinuous = False
        if e.cell_type != ct:
            raise ValueError("Enriched elements on different cell types not supported.")
        if e.polyset_type != ptype:
            raise ValueError("Enriched elements on different polyset types not supported.")
        if e.reference_value_shape != vshape or e.reference_value_size != vsize:
            raise ValueError("Enriched elements on different value shapes not supported.")
    nderivs = max(e.interpolation_nderivs for e in elements)

    x = []
    for pts_lists in zip(*[e._x for e in elements]):
        x.append([np.concatenate(pts) for pts in zip(*pts_lists)])
    M = []
    for M_lists in zip(*[e._M for e in elements]):
        M_row = []
        for M_parts in zip(*M_lists):
            ndofs = sum(mat.shape[0] for mat in M_parts)
            npts = sum(mat.shape[2] for mat in M_parts)
            deriv_dim = max(mat.shape[3] for mat in M_parts)
            new_M = np.zeros((ndofs, vsize, npts, deriv_dim))
            pt = 0
            dof = 0
            for mat in M_parts:
                new_M[dof : dof + mat.shape[0], :, pt : pt + mat.shape[2], : mat.shape[3]] = mat
                dof += mat.shape[0]
                pt += mat.shape[2]
            M_row.append(new_M)
        M.append(M_row)

    dim = sum(e.dim for e in elements)
    wcoeffs = np.zeros(
        (dim, _basix.polynomials.dim(_basix.PolynomialType.legendre, ct, hd) * vsize)
    )
    row = 0
    for e in elements:
        wcoeffs[row : row + e.dim, :] = _basix.polynomials.reshape_coefficients(
            _basix.PolynomialType.legendre, ct, e._wcoeffs, vsize, e.embedded_superdegree, hd
        )
        row += e.dim

    return custom_element(
        ct,
        list(vshape),
        wcoeffs,
        x,
        M,
        nderivs,
        map_type,
        ss,
        discontinuous,
        hcd,
        hd,
        ptype,
    )


def custom_element(
    cell_type: _basix.CellType,
    reference_value_shape: _typing.Union[list[int], tuple[int, ...]],
    wcoeffs: _npt.NDArray[np.float64],
    x: list[list[_npt.NDArray[np.float64]]],
    M: list[list[_npt.NDArray[np.float64]]],
    interpolation_nderivs: int,
    map_type: _basix.MapType,
    sobolev_space: _basix.SobolevSpace,
    discontinuous: bool,
    embedded_subdegree: int,
    embedded_superdegree: int,
    polyset_type: _basix.PolysetType = _basix.PolysetType.standard,
) -> _ElementBase:
    """Create a UFL compatible custom Basix element.

    Args:
        cell_type: The cell type
        reference_value_shape: The reference value shape of the element
        wcoeffs: Matrices for the kth value index containing the
            expansion coefficients defining a polynomial basis spanning
            the polynomial space for this element. Shape is
            ``(dim(finite element polyset), dim(Legenre polynomials))``.
        x: Interpolation points. Indices are ``(tdim, entity index,
            point index, dim)``.
        M: The interpolation matrices. Indices are ``(tdim, entity
            index, dof, vs, point_index, derivative)``.
        interpolation_nderivs: The number of derivatives that need to be
            used during interpolation.
        map_type: The type of map to be used to map values from the
            reference to a cell.
        sobolev_space: Underlying Sobolev space for the element.
        discontinuous: Indicates whether or not this is the
            discontinuous version of the element.
        embedded_subdegree: The highest degree ``n`` such that a
            Lagrange (or vector Lagrange) element of degree ``n`` is a
            subspace of this element.
        embedded_superdegree: The highest degree of a polynomial in this
            element's polyset.
        polyset_type: Polyset type for the element.

    Returns:
        A custom finite element.
    """
    e = _basix.create_custom_element(
        cell_type,
        tuple(reference_value_shape),
        wcoeffs,
        x,
        M,
        interpolation_nderivs,
        map_type,
        sobolev_space,
        discontinuous,
        embedded_subdegree,
        embedded_superdegree,
        polyset_type,
    )
    return _BasixElement(e)


def mixed_element(elements: list[_ElementBase]) -> _ElementBase:
    """Create a UFL compatible mixed element from a list of elements.

    Args:
        elements: The list of elements

    Returns:
        A mixed finite element.
    """
    return _MixedElement(elements)


def quadrature_element(
    cell: _typing.Union[str, _basix.CellType],
    value_shape: tuple[int, ...] = (),
    scheme: _typing.Optional[str] = None,
    degree: _typing.Optional[int] = None,
    points: _typing.Optional[_npt.NDArray[np.float64]] = None,
    weights: _typing.Optional[_npt.NDArray[np.float64]] = None,
    pullback: _AbstractPullback = _ufl.identity_pullback,
) -> _ElementBase:
    """Create a quadrature element.

    When creating this element, either the quadrature scheme and degree
    must be input or the quadrature points and weights must be.

    Args:
        cell: Cell to create the element on.
        value_shape: Value shape of the element.
        scheme: Quadrature scheme.
        degree: Quadrature degree.
        points: Quadrature points.
        weights: Quadrature weights.
        pullback: Map name.

    Returns:
        A 'quadrature' finite element.
    """
    if isinstance(cell, str):
        cell = _basix.cell.string_to_type(cell)

    if points is None:
        assert weights is None
        assert degree is not None
        if scheme is None:
            points, weights = _basix.make_quadrature(cell, degree)
        else:
            points, weights = _basix.make_quadrature(
                cell, degree, rule=_basix.quadrature.string_to_type(scheme)
            )

    assert points is not None
    assert weights is not None

    e = _QuadratureElement(cell, points, weights, pullback, degree)
    if value_shape == ():
        return e
    else:
        return _BlockedElement(e, value_shape)


def real_element(
    cell: _typing.Union[_basix.CellType, str], value_shape: tuple[int, ...]
) -> _ElementBase:
    """Create a real element.

    Args:
        cell: Cell to create the element on.
        value_shape: Value shape of the element.

    Returns:
        A 'real' finite element.

    """
    if isinstance(cell, str):
        cell = _basix.cell.string_to_type(cell)

    return _RealElement(cell, value_shape)


@_functools.cache
def blocked_element(
    sub_element: _ElementBase,
    shape: tuple[int, ...],
    symmetry: _typing.Optional[bool] = None,
) -> _ElementBase:
    """Create a UFL compatible blocked element.

    Args:
        sub_element: Element used for each block.
        shape: Value shape of the element. For scalar-valued families,
            this can be used to create vector and tensor elements.
        symmetry: Set to ``True`` if the tensor is symmetric. Valid for
            rank 2 elements only.

    Returns:
        A blocked finite element.
    """
    if len(sub_element.reference_value_shape) != 0:
        raise ValueError("Cannot create a blocked element containing a non-scalar element.")

    return _BlockedElement(sub_element, shape=shape, symmetry=symmetry)


def wrap_element(element: _basix.finite_element.FiniteElement) -> _ElementBase:
    """Wrap a Basix element as a Basix UFL element."""
    return _BasixElement(element)
