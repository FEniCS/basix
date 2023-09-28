"""Functions to directly wrap Basix elements in UFL."""

import functools as _functools
import hashlib as _hashlib
import itertools as _itertools
import typing as _typing
from abc import abstractmethod as _abstractmethod
from warnings import warn as _warn

import numpy as _np
import numpy.typing as _npt
import ufl as _ufl
# TODO: remove gdim arguments once UFL handles cells better
from ufl.finiteelement import FiniteElementBase as _FiniteElementBase

import basix as _basix

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


class _ElementBase(_FiniteElementBase):
    """A base wrapper to allow elements to be used with UFL.

    This class includes methods and properties needed by UFL and FFCx. This is a base class containing
    functions common to all the element types defined in this file.
    """

    def __init__(self, repr: str, name: str, cellname: str, value_shape: _typing.Tuple[int, ...],
                 degree: int = -1, mapname: _typing.Optional[str] = None,
                 gdim: _typing.Optional[int] = None):
        """Initialise the element."""
        super().__init__(name, _ufl.cell.Cell(cellname, gdim), degree, None, value_shape, value_shape)
        self._repr = repr
        self._map = mapname
        self._degree = degree
        self._value_shape = value_shape

    def __repr__(self):
        """Get the representation of the element."""
        return self._repr

    def sub_elements(self) -> _typing.List:
        """Return a list of sub elements."""
        return []

    def num_sub_elements(self) -> int:
        """Return a list of sub elements."""
        return len(self.sub_elements())

    def mapping(self) -> _typing.Union[str, None]:
        """Return the map type."""
        return self._map

    @_abstractmethod
    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        pass

    def __hash__(self) -> int:
        """Return a hash."""
        return hash("basix" + self._repr)

    @property
    def value_size(self) -> int:
        """Value size of the element."""
        vs = 1
        for i in self._value_shape:
            vs *= i
        return vs

    @property
    def reference_value_size(self) -> int:
        """Reference value size of the element."""
        vs = 1
        for i in self.reference_value_shape():
            vs *= i
        return vs

    def value_shape(self) -> _typing.Tuple[int, ...]:
        """Value shape of the element basis function."""
        return self._value_shape

    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Reference value shape of the element basis function."""
        return self._value_shape

    @property
    def block_size(self) -> int:
        """The block size of the element."""
        return 1

    def degree(self) -> int:
        """The degree of the element."""
        return self._degree

    @_abstractmethod
    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        pass

    @_abstractmethod
    def get_component_element(self, flat_component: int) -> _typing.Tuple[_typing.Any, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

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
        pass

    @property
    @_abstractmethod
    def ufcx_element_type(self) -> str:
        """Element type."""
        pass

    @property
    @_abstractmethod
    def dim(self) -> int:
        """Number of DOFs the element has."""
        pass

    @property
    @_abstractmethod
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        pass

    @property
    @_abstractmethod
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with each entity."""
        pass

    @property
    @_abstractmethod
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        pass

    @property
    @_abstractmethod
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        pass

    @property
    @_abstractmethod
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        pass

    @property
    @_abstractmethod
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        pass

    @property
    @_abstractmethod
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
        """Geometry of the reference element."""
        pass

    @property
    @_abstractmethod
    def family_name(self) -> str:
        """Family name of the element."""
        pass

    @property
    @_abstractmethod
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        pass

    @property
    @_abstractmethod
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        pass

    @property
    @_abstractmethod
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        pass

    @property
    @_abstractmethod
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        pass

    @property
    @_abstractmethod
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        pass

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

    def custom_quadrature(self) -> _typing.Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
        """True if the element has a custom quadrature rule."""
        raise ValueError("Element does not have a custom quadrature rule.")

    @property
    @_abstractmethod
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        pass

    @property
    def highest_complete_degree(self) -> int:
        """The highest complete degree of the element."""
        raise NotImplementedError()

    @property
    def highest_degree(self) -> int:
        """The highest degree of the element."""
        raise NotImplementedError()

    @property
    @_abstractmethod
    def polyset_type(self) -> _basix.PolysetType:
        """The polyset type of the element."""

    @property
    def _wcoeffs(self) -> _npt.NDArray[_np.float64]:
        """The coefficients used to define the polynomial set."""
        raise NotImplementedError()

    @property
    def _x(self) -> _typing.List[_typing.List[_npt.NDArray[_np.float64]]]:
        """The points used to define interpolation."""
        raise NotImplementedError()

    @property
    def _M(self) -> _typing.List[_typing.List[_npt.NDArray[_np.float64]]]:
        """The matrices used to define interpolation."""
        raise NotImplementedError()

    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be computed
        as a tensor product of the basis elements of the elements in the factoriaation.
        """
        return False

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        return None

    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return _ufl_sobolev_space_from_enum(self.basix_sobolev_space)

    @property
    @_abstractmethod
    def basix_sobolev_space(self):
        """Return a Basix enum representing the underlying Sobolev space."""
        pass


class _BasixElement(_ElementBase):
    """A wrapper allowing Basix elements to be used directly with UFL.

    This class allows elements created with `basix.create_element` to be wrapped as UFL compatible elements.
    Users should not directly call this class's initiliser, but should use the `element` function instead.
    """

    element: _basix.finite_element.FiniteElement

    def __init__(self, element: _basix.finite_element.FiniteElement, gdim: _typing.Optional[int] = None):
        """Create a Basix element."""
        if element.family == _basix.ElementFamily.custom:
            self._is_custom = True
            repr = f"custom Basix element ({_compute_signature(element)})"
        else:
            self._is_custom = False
            repr = (f"Basix element ({element.family.name}, {element.cell_type.name}, {element.degree}, "
                    f"{element.lagrange_variant.name}, {element.dpc_variant.name}, {element.discontinuous})")

        super().__init__(
            repr, element.family.name, element.cell_type.name, tuple(element.value_shape), element.degree,
            _map_type_to_string(element.map_type), gdim=gdim)

        self.element = element

    @property
    def basix_sobolev_space(self):
        """Return a Basix enum representing the underlying Sobolev space."""
        return self.element.sobolev_space

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, _BasixElement) and self.element == other.element

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        tab = self.element.tabulate(nderivs, points)
        # TODO: update FFCx to remove the need for transposing here
        return tab.transpose((0, 1, 3, 2)).reshape((tab.shape[0], tab.shape[1], -1))

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_ElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

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
        assert flat_component < self.value_size
        return _ComponentElement(self, flat_component), 0, 1

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
        return self.element.dim

    @property
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        return self.element.num_entity_dofs

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with each entity."""
        return self.element.entity_dofs

    @property
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return self.element.num_entity_closure_dofs

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        return self.element.entity_closure_dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return 0

    @property
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        return _basix.topology(self.element.cell_type)

    @property
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
        """Geometry of the reference element."""
        return _basix.geometry(self.element.cell_type)

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return self.element.family.name

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return self.element.family

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return self.element.lagrange_variant

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return self.element.dpc_variant

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self.element.cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self.element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self.element.interpolation_nderivs

    @property
    def is_custom_element(self) -> bool:
        """True if the element is a custom Basix element."""
        return self._is_custom

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        return self.element.map_type

    @property
    def highest_complete_degree(self) -> int:
        """The highest complete degree of the element."""
        return self.element.highest_complete_degree

    @property
    def highest_degree(self) -> int:
        """The highest degree of the element."""
        return self.element.highest_degree

    @property
    def polyset_type(self) -> _basix.PolysetType:
        return self.element.polyset_type

    @property
    def _wcoeffs(self) -> _npt.NDArray[_np.float64]:
        """The coefficients used to define the polynomial set."""
        return self.element.wcoeffs

    @property
    def _x(self) -> _typing.List[_typing.List[_npt.NDArray[_np.float64]]]:
        """The points used to define interpolation."""
        return self.element.x

    @property
    def _M(self) -> _typing.List[_typing.List[_npt.NDArray[_np.float64]]]:
        """The matrices used to define interpolation."""
        return self.element.M

    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be
        computed as a tensor product of the basis elements of the
        elements in the factorisation.

        """
        return self.element.has_tensor_product_factorisation

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        if not self.has_tensor_product_factorisation:
            return None
        return self.element.get_tensor_product_representation()


class _ComponentElement(_ElementBase):
    """An element representing one component of a _BasixElement.

    This element type is used when UFL's ``get_component_element``
    function is called.

    """
    element: _ElementBase
    component: int

    def __init__(self, element: _ElementBase, component: int, gdim: _typing.Optional[int] = None):
        """Initialise the element."""
        self.element = element
        self.component = component
        super().__init__(f"component element ({element._repr}, {component})",
                         f"Component of {element.family_name}",
                         element.cell_type.name, (1, ), element._degree, gdim=gdim)

    @property
    def basix_sobolev_space(self):
        """Return a Basix enum representing the underlying Sobolev space."""
        return self.element.basix_sobolev_space

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (isinstance(other, _ComponentElement) and self.element == other.element
                and self.component == other.component)

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at.

        Returns:
            Tabulated basis functions.

        """
        tables = self.element.tabulate(nderivs, points)
        output = []
        for tbl in tables:
            shape = (tbl.shape[0],) + self.element._value_shape + (-1,)
            tbl = tbl.reshape(shape)
            if len(self.element._value_shape) == 0:
                output.append(tbl)
            elif len(self.element._value_shape) == 1:
                output.append(tbl[:, self.component, :])
            elif len(self.element._value_shape) == 2:
                if isinstance(self.element, _BlockedElement) and self.element._has_symmetry:
                    # FIXME: check that this behaves as expected
                    output.append(tbl[:, self.component, :])
                else:
                    vs0 = self.element._value_shape[0]
                    output.append(tbl[:, self.component // vs0, self.component % vs0, :])
            else:
                raise NotImplementedError()
        return _np.asarray(output, dtype=_np.float64)

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_ElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        if flat_component == 0:
            return self, 0, 1
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        raise NotImplementedError()

    @property
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        raise NotImplementedError()

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with each entity."""
        raise NotImplementedError()

    @property
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        raise NotImplementedError()

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
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
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
        """Geometry of the reference element."""
        raise NotImplementedError()

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return self.element.element_family

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return self.element.lagrange_variant

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return self.element.dpc_variant

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self.element.cell_type

    @property
    def polyset_type(self) -> _basix.PolysetType:
        return self.element.polyset_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self.element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self.element.interpolation_nderivs

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        raise NotImplementedError()

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        raise NotImplementedError()

    def __mul__(self, other):
        _warn("Use of * to create mixed elements is deprecated and will be removed after December 2023. "
              "Please, use basix.ufl.mixed_element.", FutureWarning)
        return mixed_element([self, other])


class _MixedElement(_ElementBase):
    """A mixed element that combines two or more elements.

    This can be used when multiple different elements appear in a form.
    Users should not directly call this class's initilizer, but should
    use the :func:`mixed_element` function instead.

    """

    _sub_elements: _typing.List[_ElementBase]

    def __init__(self, sub_elements: _typing.List[_ElementBase], gdim: _typing.Optional[int] = None):
        """Initialise the element."""
        assert len(sub_elements) > 0
        self._sub_elements = sub_elements
        if all(e.mapping() == "identity" for e in sub_elements):
            mapname = "identity"
        else:
            mapname = "undefined"

        super().__init__("mixed element (" + ", ".join(i._repr for i in sub_elements) + ")",
                         "mixed element", sub_elements[0].cell_type.name,
                         (sum(i.value_size for i in sub_elements), ), mapname=mapname, gdim=gdim)

    def degree(self) -> int:
        """Degree of the element."""
        return max((e.degree() for e in self._sub_elements), default=-1)

    @property
    def map_type(self) -> _basix.MapType:
        """Basix map type."""
        raise NotImplementedError()

    @property
    def basix_sobolev_space(self):
        """Basix Sobolev space that the element belongs to."""
        return _basix.sobolev_spaces.intersection([e.basix_sobolev_space for e in self._sub_elements])

    def sub_elements(self) -> _typing.List[_ElementBase]:
        """List of sub elements."""
        return self._sub_elements

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

    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
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
            new_table = _np.zeros((len(points), self.value_size * self.dim))
            start = 0
            for e, t in zip(self._sub_elements, deriv_tables):
                for i in range(0, e.dim, e.value_size):
                    new_table[:, start: start + e.value_size] = t[:, i: i + e.value_size]
                    start += self.value_size
            tables.append(new_table)
        return _np.asarray(tables, dtype=_np.float64)

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_ElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        sub_dims = [0] + [e.dim for e in self._sub_elements]
        sub_cmps = [0] + [e.value_size for e in self._sub_elements]

        irange = _np.cumsum(sub_dims)
        crange = _np.cumsum(sub_cmps)

        # Find index of sub element which corresponds to the current
        # flat component
        component_element_index = _np.where(crange <= flat_component)[0].shape[0] - 1

        sub_e = self._sub_elements[component_element_index]

        e, offset, stride = sub_e.get_component_element(flat_component - crange[component_element_index])
        # TODO: is this offset correct?
        return e, irange[component_element_index] + offset, stride

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return "ufcx_mixed_element"

    @property
    def dim(self) -> int:
        """Dimension (number of DOFs) for the element."""
        return sum(e.dim for e in self._sub_elements)

    @property
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        data = [e.num_entity_dofs for e in self._sub_elements]
        return [[sum(d[tdim][entity_n] for d in data) for entity_n, _ in enumerate(entities)]
                for tdim, entities in enumerate(data[0])]

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with each entity."""
        dofs: _typing.List[_typing.List[_typing.List[int]]] = [
            [[] for i in entities]
            for entities in self._sub_elements[0].entity_dofs]
        start_dof = 0
        for e in self._sub_elements:
            for tdim, entities in enumerate(e.entity_dofs):
                for entity_n, entity_dofs in enumerate(entities):
                    dofs[tdim][entity_n] += [start_dof + i for i in entity_dofs]
            start_dof += e.dim
        return dofs

    @property
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        data = [e.num_entity_closure_dofs for e in self._sub_elements]
        return [[sum(d[tdim][entity_n] for d in data) for entity_n, _ in enumerate(entities)]
                for tdim, entities in enumerate(data[0])]

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        dofs: _typing.List[_typing.List[_typing.List[int]]] = [
            [[] for i in entities]
            for entities in self._sub_elements[0].entity_closure_dofs]
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
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        return self._sub_elements[0].reference_topology

    @property
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
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


class _BlockedElement(_ElementBase):
    """Element with a block size that contains multiple copies of a sub element.

    This can be used to (for example) create vector and tensor Lagrange
    elements. Users should not directly call this classes initilizer,
    but should use the `blocked_element` function instead.

    """
    block_shape: _typing.Tuple[int, ...]
    sub_element: _ElementBase
    _block_size: int

    def __init__(self, sub_element: _ElementBase, shape: _typing.Tuple[int, ...],
                 symmetry: _typing.Optional[bool] = None, gdim: _typing.Optional[int] = None,):
        """Initialise the element."""
        if sub_element.value_size != 1:
            raise ValueError("Blocked elements of non-scalar elements are not supported. "
                             "Try using _MixedElement instead.")
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

        self.sub_element = sub_element
        self._block_size = block_size
        self.block_shape = shape

        repr = f"blocked element ({sub_element._repr}, {shape}"
        if len(shape) == 2:
            if symmetry:
                repr += ", True"
            else:
                repr += ", False"
        repr += ")"

        super().__init__(repr, sub_element.family(), sub_element.cell_type.name, shape,
                         sub_element._degree, sub_element._map, gdim=gdim)

        if symmetry:
            n = 0
            sub_element_mapping = {}
            for i in range(shape[0]):
                for j in range(i + 1):
                    sub_element_mapping[(i, j)] = n
                    sub_element_mapping[(j, i)] = n
                    n += 1

            self._map = "symmetries"
            self._symmetry = {(i, j): (j, i) for i in range(shape[0]) for j in range(i)}
            self._flattened_sub_element_mapping = [
                sub_element_mapping[(i, j)] for i in range(shape[0]) for j in range(shape[1])]

    @property
    def basix_sobolev_space(self):
        """Basix enum representing the underlying Sobolev space."""
        return self.sub_element.basix_sobolev_space

    def sub_elements(self) -> _typing.List[_ElementBase]:
        """List of sub elements."""
        return [self.sub_element for _ in range(self._block_size)]

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (
            isinstance(other, _BlockedElement) and self._block_size == other._block_size
            and self.block_shape == other.block_shape and self.sub_element == other.sub_element)

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    @property
    def block_size(self) -> int:
        """Block size of the element."""
        return self._block_size

    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Reference value shape of the element basis function."""
        if self._has_symmetry:
            assert len(self.block_shape) == 2 and self.block_shape[0] == self.block_shape[1]
            return (self.block_shape[0] * (self.block_shape[0] + 1) // 2, )
        return self._value_shape

    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        assert len(self.block_shape) == 1  # TODO: block shape
        assert self.value_size == self._block_size  # TODO: remove this assumption
        output = []
        for table in self.sub_element.tabulate(nderivs, points):
            # Repeat sub element horizontally
            assert len(table.shape) == 2
            new_table = _np.zeros((table.shape[0], *self.block_shape,
                                   self._block_size * table.shape[1]))
            for i, j in enumerate(_itertools.product(*[range(s) for s in self.block_shape])):
                if len(j) == 1:
                    new_table[:, j[0], i::self._block_size] = table
                elif len(j) == 2:
                    new_table[:, j[0], j[1], i::self._block_size] = table
                else:
                    raise NotImplementedError()
            output.append(new_table)
        return _np.asarray(output, dtype=_np.float64)

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_ElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        return self.sub_element, flat_component, self._block_size

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return self.sub_element.ufcx_element_type

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        return self.sub_element.dim * self._block_size

    @property
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        return [[j * self._block_size for j in i] for i in self.sub_element.num_entity_dofs]

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with each entity."""
        # TODO: should this return this, or should it take blocks into
        # account?
        return [[[k * self._block_size + b for k in j for b in range(self._block_size)]
                 for j in i] for i in self.sub_element.entity_dofs]

    @property
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return [[j * self._block_size for j in i] for i in self.sub_element.num_entity_closure_dofs]

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        # TODO: should this return this, or should it take blocks into
        # account?
        return [[[k * self._block_size + b for k in j for b in range(self._block_size)]
                 for j in i] for i in self.sub_element.entity_closure_dofs]

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return self.sub_element.num_global_support_dofs * self._block_size

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return self.sub_element.family_name

    @property
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        return self.sub_element.reference_topology

    @property
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
        """Geometry of the reference element."""
        return self.sub_element.reference_geometry

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        return self.sub_element.lagrange_variant

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        return self.sub_element.dpc_variant

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        return self.sub_element.element_family

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        return self.sub_element.cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self.sub_element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self.sub_element.interpolation_nderivs

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        return self.sub_element.map_type

    @property
    def highest_complete_degree(self) -> int:
        """The highest complete degree of the element."""
        return self.sub_element.highest_complete_degree

    @property
    def highest_degree(self) -> int:
        """The highest degree of the element."""
        return self.sub_element.highest_degree

    @property
    def polyset_type(self) -> _basix.PolysetType:
        return self.sub_element.polyset_type

    @property
    def _wcoeffs(self) -> _npt.NDArray[_np.float64]:
        """Coefficients used to define the polynomial set."""
        sub_wc = self.sub_element._wcoeffs
        wcoeffs = _np.zeros((sub_wc.shape[0] * self._block_size, sub_wc.shape[1] * self.block_size))
        for i in range(self._block_size):
            wcoeffs[sub_wc.shape[0] * i: sub_wc.shape[0]
                    * (i + 1), sub_wc.shape[1] * i: sub_wc.shape[1] * (i + 1)] = sub_wc
        return wcoeffs

    @property
    def _x(self) -> _typing.List[_typing.List[_npt.NDArray[_np.float64]]]:
        """Points used to define interpolation."""
        return self.sub_element._x

    @property
    def _M(self) -> _typing.List[_typing.List[_npt.NDArray[_np.float64]]]:
        """Matrices used to define interpolation."""
        M = []
        for M_list in self.sub_element._M:
            M_row = []
            for mat in M_list:
                new_mat = _np.zeros((mat.shape[0] * self._block_size, mat.shape[1]
                                     * self._block_size, mat.shape[2], mat.shape[3]))
                for i in range(self._block_size):
                    new_mat[i * mat.shape[0]: (i + 1) * mat.shape[0],
                            i * mat.shape[1]: (i + 1) * mat.shape[1], :, :] = mat
                M_row.append(new_mat)
            M.append(M_row)
        return M

    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be
        computed as a tensor product of the basis elements of the
        elements in the factoriaation.

        """
        return self.sub_element.has_tensor_product_factorisation()

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        if not self.has_tensor_product_factorisation:
            return None
        return self.sub_element.get_tensor_product_representation()

    def flattened_sub_element_mapping(self) -> _typing.List[int]:
        """Return the flattened sub element mapping."""
        if not self._has_symmetry:
            raise ValueError("Cannot get flattened map for non-symmetric element.")
        return self._flattened_sub_element_mapping


class _QuadratureElement(_ElementBase):
    """A quadrature element."""

    def __init__(self, cell: _basix.CellType, value_shape: _typing.Tuple[int, ...],
                 points: _npt.NDArray[_np.float64], weights: _npt.NDArray[_np.float64],
                 mapname: str, degree: _typing.Optional[int] = None):
        """Initialise the element."""
        self._points = points
        self._weights = weights
        repr = f"QuadratureElement({cell.name}, {points!r}, {weights!r}, {mapname})".replace("\n", "")
        self._cell_type = cell
        self._entity_counts = [len(i) for i in _basix.topology(cell)]

        if degree is None:
            degree = len(points)

        super().__init__(repr, "quadrature element", cell.name, value_shape, degree, mapname=mapname)

    def basix_sobolev_space(self):
        """Return the underlying Sobolev space."""
        return _basix.sobolev_spaces.L2

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, _QuadratureElement) and _np.allclose(self._points, other._points) and \
            _np.allclose(self._weights, other._weights)

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
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
        tables = _np.asarray([_np.eye(points.shape[0], points.shape[0])])
        return tables

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_ElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        return self, 0, 1

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
        return "ufcx_quadrature_element"

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
        return self._points.shape[0]

    @property
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        dofs = []
        for d in self._entity_counts[:-1]:
            dofs += [[0] * d]

        dofs += [[self.dim]]
        return dofs

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
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
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return self.num_entity_dofs

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        return self.entity_dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return 0

    @property
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
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

    def custom_quadrature(self) -> _typing.Tuple[_npt.NDArray[_np.float64], _npt.NDArray[_np.float64]]:
        """True if the element has a custom quadrature rule."""
        return self._points, self._weights


class _RealElement(_ElementBase):
    """A real element."""

    def __init__(self, cell: _basix.CellType, value_shape: _typing.Tuple[int, ...]):
        """Initialise the element."""
        self._cell_type = cell
        tdim = len(_basix.topology(cell)) - 1

        super().__init__(
            f"RealElement({element})", "real element", cell.name, value_shape, 0)

        self._entity_counts = []
        if tdim >= 1:
            self._entity_counts.append(self.cell().num_vertices())
        if tdim >= 2:
            self._entity_counts.append(self.cell().num_edges())
        if tdim >= 3:
            self._entity_counts.append(self.cell().num_facets())
        self._entity_counts.append(1)

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, _RealElement)

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(self, nderivs: int, points: _npt.NDArray[_np.float64]) -> _npt.NDArray[_np.float64]:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions

        """
        out = _np.zeros((nderivs + 1, len(points), self.value_size**2))
        for v in range(self.value_size):
            out[0, :, self.value_size * v + v] = 1.
        return out

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_ElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component

        """
        assert flat_component < self.value_size
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
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        dofs = []
        for d in self._entity_counts[:-1]:
            dofs += [[0] * d]

        dofs += [[self.dim]]
        return dofs

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
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
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        return self.num_entity_dofs

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        return self.entity_dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return 1

    @property
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _npt.NDArray[_np.float64]:
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

    def basix_sobolev_space(self):
        """Return the underlying Sobolev space."""
        return _basix.sobolev_spaces.Hinf

    @property
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        return _basix.MapType.identity

    @property
    def polyset_type(self) -> _basix.PolysetType:
        """The polyset type of the element."""
        raise NotImplementedError()


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
    signature += _hashlib.sha1(data.encode('utf-8')).hexdigest()

    return signature


@_functools.lru_cache()
def element(family: _typing.Union[_basix.ElementFamily, str], cell: _typing.Union[_basix.CellType, str], degree: int,
            lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
            dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset, discontinuous: bool = False,
            shape: _typing.Optional[_typing.Tuple[int, ...]] = None,
            symmetry: _typing.Optional[bool] = None, gdim: _typing.Optional[int] = None) -> _ElementBase:
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
        gdim: Geometric dimension. If not set the geometric dimension is
            set equal to the topological dimension of the cell.

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
            _warn("\"CG\" element name is deprecated. Consider using \"Lagrange\" or \"P\" instead",
                  DeprecationWarning, stacklevel=2)
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

    e = _basix.create_element(family, cell, degree, lagrange_variant, dpc_variant, discontinuous)
    ufl_e = _BasixElement(e, gdim=gdim)

    if shape is None or shape == tuple(e.value_shape):
        if symmetry is not None:
            raise ValueError("Cannot pass a symmetry argument to this element.")
        return ufl_e
    else:
        return blocked_element(ufl_e, shape=shape, gdim=gdim, symmetry=symmetry)


def enriched_element(elements: _typing.List[_ElementBase],
                     map_type: _typing.Optional[_basix.MapType] = None,
                     gdim: _typing.Optional[int] = None) -> _ElementBase:
    """Create an UFL compatible enriched element from a list of elements.

    Args:
        elements: The list of elements
        map_type: The map type for the enriched element.
        gdim: Geometric dimension. If not set the geometric dimension is
            set equal to the topological dimension of the cell.

    Returns:
        An enriched finite element.

    """
    ct = elements[0].cell_type
    ptype = elements[0].polyset_type
    vshape = elements[0].value_shape()
    vsize = elements[0].value_size
    if map_type is None:
        map_type = elements[0].map_type
        for e in elements:
            if e.map_type != map_type:
                raise ValueError("Enriched elements on different map types not supported.")

    hcd = min(e.highest_complete_degree for e in elements)
    hd = max(e.highest_degree for e in elements)
    ss = _basix.sobolev_spaces.intersection([e.basix_sobolev_space for e in elements])
    discontinuous = True
    for e in elements:
        if not e.discontinuous:
            discontinuous = False
        if e.cell_type != ct:
            raise ValueError("Enriched elements on different cell types not supported.")
        if e.polyset_type != ptype:
            raise ValueError("Enriched elements on different polyset types not supported.")
        if e.value_shape() != vshape or e.value_size != vsize:
            raise ValueError("Enriched elements on different value shapes not supported.")
    nderivs = max(e.interpolation_nderivs for e in elements)

    x = []
    for pts_lists in zip(*[e._x for e in elements]):
        x.append([_np.concatenate(pts) for pts in zip(*pts_lists)])
    M = []
    for M_lists in zip(*[e._M for e in elements]):
        M_row = []
        for M_parts in zip(*M_lists):
            ndofs = sum(mat.shape[0] for mat in M_parts)
            npts = sum(mat.shape[2] for mat in M_parts)
            deriv_dim = max(mat.shape[3] for mat in M_parts)
            new_M = _np.zeros((ndofs, vsize, npts, deriv_dim))
            pt = 0
            dof = 0
            for i, mat in enumerate(M_parts):
                new_M[dof: dof + mat.shape[0], :, pt: pt + mat.shape[2], :mat.shape[3]] = mat
                dof += mat.shape[0]
                pt += mat.shape[2]
            M_row.append(new_M)
        M.append(M_row)

    dim = sum(e.dim for e in elements)
    wcoeffs = _np.zeros((dim, _basix.polynomials.dim(_basix.PolynomialType.legendre, ct, hd) * vsize))
    row = 0
    for e in elements:
        wcoeffs[row: row + e.dim, :] = _basix.polynomials.reshape_coefficients(
            _basix.PolynomialType.legendre, ct, e._wcoeffs, vsize, e.highest_degree, hd)
        row += e.dim

    return custom_element(ct, list(vshape), wcoeffs, x, M, nderivs,
                          map_type, ss, discontinuous, hcd, hd, ptype, gdim=gdim)


def custom_element(cell_type: _basix.CellType, value_shape: _typing.Union[_typing.List[int], _typing.Tuple[int, ...]],
                   wcoeffs: _npt.NDArray[_np.float64], x: _typing.List[_typing.List[_npt.NDArray[_np.float64]]],
                   M: _typing.List[_typing.List[_npt.NDArray[_np.float64]]], interpolation_nderivs: int,
                   map_type: _basix.MapType, sobolev_space: _basix.SobolevSpace, discontinuous: bool,
                   highest_complete_degree: int, highest_degree: int,
                   polyset_type: _basix.PolysetType = _basix.PolysetType.standard,
                   gdim: _typing.Optional[int] = None) -> _ElementBase:
    """Create a UFL compatible custom Basix element.

    Args:
        cell_type: The cell type
        value_shape: The value shape of the element
        wcoeffs: Matrices for the kth value index containing the
            expansion coefficients defining a polynomial basis spanning
            the polynomial space for this element. Shape is
            ``(dim(finite element polyset), dim(Legenre polynomials))``.
        x: Interpolation points. Indices are ``(tdim, entity index,
            point index, dim)``
        M: The interpolation matrices. Indices are ``(tdim, entity
            index, dof, vs, point_index, derivative)``.
        interpolation_nderivs: The number of derivatives that need to be
            used during interpolation.
        map_type: The type of map to be used to map values from the
            reference to a cell.
        sobolev_space: Underlying Sobolev space for the element.
        discontinuous: Indicates whether or not this is the
            discontinuous version of the element.
        highest_complete_degree: The highest degree ``n`` such that a
            Lagrange (or vector Lagrange) element of degree ``n`` is a
            subspace of this element.
        highest_degree: The degree of a polynomial in this element's
            polyset.
        polyset_type: Polyset type for the element.
        gdim: Geometric dimension. If not set the geometric dimension is
            set equal to the topological dimension of the cell.

    Returns:
        A custom finite element.

    """
    return _BasixElement(_basix.create_custom_element(
        cell_type, list(value_shape), wcoeffs, x, M, interpolation_nderivs,
        map_type, sobolev_space, discontinuous, highest_complete_degree,
        highest_degree, polyset_type), gdim=gdim)


def mixed_element(elements: _typing.List[_ElementBase], gdim: _typing.Optional[int] = None) -> _ElementBase:
    """Create a UFL compatible mixed element from a list of elements.

    Args:
        elements: The list of elements
        gdim: Geometric dimension. If not set the geometric dimension is
            set equal to the topological dimension of the cell.

    Returns:
        A mixed finite element.

    """
    return _MixedElement(elements, gdim=gdim)


def quadrature_element(cell: _typing.Union[str, _basix.CellType],
                       value_shape: _typing.Tuple[int, ...],
                       scheme: _typing.Optional[str] = None,
                       degree: _typing.Optional[int] = None,
                       points: _typing.Optional[_npt.NDArray[_np.float64]] = None,
                       weights: _typing.Optional[_npt.NDArray[_np.float64]] = None,
                       mapname: str = "identity") -> _ElementBase:
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
        mapname: Map name.

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
                cell, degree, rule=_basix.quadrature.string_to_type(scheme))

    assert points is not None
    assert weights is not None

    return _QuadratureElement(cell, value_shape, points, weights, mapname, degree)


def real_element(cell: _typing.Union[_basix.CellType, str],
                 value_shape: _typing.Tuple[int, ...]) -> _ElementBase:
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


@_functools.lru_cache()
def blocked_element(
    sub_element: _ElementBase, shape: _typing.Tuple[int, ...],
    symmetry: _typing.Optional[bool] = None, gdim: _typing.Optional[int] = None
) -> _ElementBase:
    """Create a UFL compatible blocked element.

    Args:
        sub_element: Element used for each block.
        shape: Value shape of the element. For scalar-valued families,
            this can be used to create vector and tensor elements.
        symmetry: Set to ``True`` if the tensor is symmetric. Valid for
            rank 2 elements only.
        gdim: Geometric dimension. If not set the geometric dimension is
            set equal to the topological dimension of the cell.

    Returns:
        A blocked finite element.

    """
    if len(sub_element.value_shape()) != 0:
        raise ValueError("Cannot create a blocked element containing a non-scalar element.")

    return _BlockedElement(sub_element, shape=shape, symmetry=symmetry, gdim=gdim)


def convert_ufl_element(ufl_element: _FiniteElementBase) -> _ElementBase:
    """Convert a UFL element to a UFL compatible Basix element."""
    _warn("Converting elements created in UFL to Basix elements is deprecated. You should create the elements directly "
          "using basix.ufl.element instead", DeprecationWarning, stacklevel=2)
    if isinstance(ufl_element, _ElementBase):
        return ufl_element
    elif hasattr(_ufl, "VectorElement") and isinstance(ufl_element, _ufl.VectorElement):
        return _BlockedElement(convert_ufl_element(ufl_element.sub_elements()[0]), (ufl_element.num_sub_elements(), ))
    elif hasattr(_ufl, "TensorElement") and isinstance(ufl_element, _ufl.TensorElement):
        return _BlockedElement(convert_ufl_element(ufl_element.sub_elements()[0]), ufl_element._value_shape,
                               symmetry=ufl_element.symmetry())
    elif hasattr(_ufl, "MixedElement") and isinstance(ufl_element, _ufl.MixedElement):
        return _MixedElement([convert_ufl_element(e) for e in ufl_element.sub_elements()])
    elif hasattr(_ufl, "EnrichedElement") and isinstance(ufl_element, _ufl.EnrichedElement):
        return enriched_element([convert_ufl_element(e) for e in ufl_element._elements])
    elif hasattr(ufl_element, "family") and callable(ufl_element.family) and ufl_element.family() == "Quadrature":
        return quadrature_element(ufl_element.cell().cellname(), ufl_element.value_shape(),
                                  scheme=ufl_element.quadrature_scheme(), degree=ufl_element.degree())
    elif hasattr(ufl_element, "family") and callable(ufl_element.family) and ufl_element.family() == "Real":
        return real_element(ufl_element.cell().cellname(), ufl_element.value_shape())
    # Elements that will not be supported
    elif hasattr(_ufl, "NodalEnrichedElement") and isinstance(ufl_element, _ufl.NodalEnrichedElement):
        raise RuntimeError("NodalEnrichedElement is not supported. Use EnrichedElement instead.")
    elif hasattr(_ufl, "BrokenElement") and isinstance(ufl_element, _ufl.BrokenElement):
        raise RuntimeError("BrokenElement not supported.")
    elif hasattr(_ufl, "HCurlElement") and isinstance(ufl_element, _ufl.HCurlElement):
        raise RuntimeError("HCurlElement not supported.")
    elif hasattr(_ufl, "HDivElement") and isinstance(ufl_element, _ufl.HDivElement):
        raise RuntimeError("HDivElement not supported.")
    elif hasattr(_ufl, "TensorProductElement") and isinstance(ufl_element, _ufl.TensorProductElement):
        raise RuntimeError("TensorProductElement not supported.")
    elif hasattr(_ufl, "RestrictedElement") and isinstance(ufl_element, _ufl.RestrictedElement):
        raise RuntimeError("RestricedElement not supported.")
    elif isinstance(ufl_element, _ufl.FiniteElementBase):
        # Create a basix element from family name, cell type and degree
        family_name = ufl_element.family()
        discontinuous = False
        if family_name.startswith("Discontinuous "):
            family_name = family_name[14:]
            discontinuous = True
        if family_name == "DP":
            family_name = "P"
            discontinuous = True
        if family_name == "DQ":
            family_name = "Q"
            discontinuous = True
        if family_name == "DPC":
            discontinuous = True

        family_type = _basix.finite_element.string_to_family(family_name, ufl_element.cell().cellname())
        cell_type = _basix.cell.string_to_type(ufl_element.cell().cellname())

        variant_info = {"lagrange_variant": _basix.LagrangeVariant.unset,
                        "dpc_variant": _basix.DPCVariant.unset}
        if family_type == _basix.ElementFamily.P and ufl_element.variant() == "equispaced":
            # This is used for elements defining cells
            variant_info["lagrange_variant"] = _basix.LagrangeVariant.equispaced
        else:
            if ufl_element.variant() is not None:
                raise ValueError("UFL variants are not supported by FFCx. Please wrap a Basix element directly.")

            EF = _basix.ElementFamily
            if family_type == EF.P:
                variant_info["lagrange_variant"] = _basix.LagrangeVariant.gll_warped
            elif family_type in [EF.RT, EF.N1E]:
                variant_info["lagrange_variant"] = _basix.LagrangeVariant.legendre
            elif family_type in [EF.serendipity, EF.BDM, EF.N2E]:
                variant_info["lagrange_variant"] = _basix.LagrangeVariant.legendre
                variant_info["dpc_variant"] = _basix.DPCVariant.legendre
            elif family_type == EF.DPC:
                variant_info["dpc_variant"] = _basix.DPCVariant.diagonal_gll

        return element(family_type, cell_type, ufl_element.degree(), **variant_info, discontinuous=discontinuous,
                       gdim=ufl_element.cell().geometric_dimension())
    else:
        raise ValueError(f"Unrecognised element type: {ufl_element.__class__.__name__}")
