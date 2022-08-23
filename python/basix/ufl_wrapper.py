"""Functions to directly wrap Basix elements in UFL."""

import ufl as _ufl
from ufl.finiteelement.finiteelementbase import FiniteElementBase as _FiniteElementBase
from ufl.sobolevspace import L2, H1, H2, HDiv, HCurl, HDivDiv, HEin

import hashlib as _hashlib
import numpy as _numpy
import numpy.typing as _numpy_typing
import basix as _basix
import typing as _typing
import functools as _functools

_nda_f64 = _numpy_typing.NDArray[_numpy.float64]


class _BasixElementBase(_FiniteElementBase):
    """A base wrapper to allow Basix elements to be used with UFL.

    This class includes methods and properties needed by UFL and FFCx.
    """

    def __init__(self, repr: str, name: str, cellname: str, value_shape: _typing.Tuple[int, ...],
                 degree: int = -1, mapname: str = None):
        """Initialise the element."""
        super().__init__(name, cellname, degree, None, value_shape, value_shape)
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

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        raise NotImplementedError()

    def __hash__(self) -> int:
        """Return a hash."""
        return hash(self._repr)

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

    def tabulate(
        self, nderivs: int, points: _nda_f64
    ) -> _nda_f64:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions
        """
        raise NotImplementedError()

    # def get_component_element(self, flat_component: int) -> _typing.Tuple[_BasixElementBase, int, int]:
    def get_component_element(self, flat_component: int) -> _typing.Tuple[_typing.Any, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        For example, for a MixedElement, this will return the
        sub-element that represents the given component, the offset of
        that sub-element, and a stride of 1. For a BlockedElement, this
        will return the sub-element, an offset equal to the component
        number, and a stride equal to the block size. For vector-valued
        element (eg H(curl) and H(div) elements), this returns a
        ComponentElement (and as offset of 0 and a stride of 1). When
        tabulate is called on the ComponentElement, only the part of the
        table for the given component is returned.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """
        raise NotImplementedError()

    @property
    def ufcx_element_type(self) -> str:
        """Element type."""
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
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        raise NotImplementedError()

    @property
    def reference_geometry(self) -> _nda_f64:
        """Geometry of the reference element."""
        raise NotImplementedError()

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        raise NotImplementedError()

    @property
    def element_family(self) -> _typing.Union[_basix.ElementFamily, None]:
        """Basix element family used to initialise the element."""
        raise NotImplementedError()

    @property
    def lagrange_variant(self) -> _typing.Union[_basix.LagrangeVariant, None]:
        """Basix Lagrange variant used to initialise the element."""
        raise NotImplementedError()

    @property
    def dpc_variant(self) -> _typing.Union[_basix.DPCVariant, None]:
        """Basix DPC variant used to initialise the element."""
        raise NotImplementedError()

    @property
    def cell_type(self) -> _basix.CellType:
        """Basix cell type used to initialise the element."""
        raise NotImplementedError()

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
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
    def map_type(self) -> _basix.MapType:
        """The Basix map type."""
        raise NotImplementedError()

    @property
    def highest_complete_degree(self) -> int:
        """The highest complete degree of the element."""
        raise NotImplementedError()

    @property
    def highest_degree(self) -> int:
        """The highest degree of the element."""
        raise NotImplementedError()

    @property
    def _wcoeffs(self) -> _nda_f64:
        """The coefficients used to define the polynomial set."""
        raise NotImplementedError()

    @property
    def _x(self) -> _typing.List[_typing.List[_nda_f64]]:
        """The points used to define interpolation."""
        raise NotImplementedError()

    @property
    def _M(self) -> _typing.List[_typing.List[_nda_f64]]:
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


class BasixElement(_BasixElementBase):
    """A wrapper allowing Basix elements to be used directly with UFL."""

    element: _basix.finite_element.FiniteElement

    def __init__(self, element: _basix.finite_element.FiniteElement, repr: str = None):
        """Create a Basix element."""
        if element.family == _basix.ElementFamily.custom:
            self._is_custom = True
            if repr is None:
                repr = f"custom Basix element ({_compute_signature(element)})"
        else:
            self._is_custom = False
            if repr is None:
                repr = (f"Basix element ({element.family.name}, {element.cell_type.name}, {element.degree}, "
                        f"{element.lagrange_variant.name}, {element.dpc_variant.name}, {element.discontinuous})")

        super().__init__(
            repr, element.family.name, element.cell_type.name, tuple(element.value_shape), element.degree,
            _map_type_to_string(element.map_type))

        self.element = element

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, BasixElement) and self.element == other.element

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        # TODO: get elements to report their Sobolev space
        EF = _basix.ElementFamily
        if self.element.discontinuous:
            return L2
        if self.element.family in [EF.P, EF.Bubble, EF.serendipity]:
            return H1
        if self.element.family in [EF.Hermite]:
            return H2
        if self.element.family in [EF.BDM, EF.RT]:
            return HDiv
        if self.element.family in [EF.N2E, EF.N1E]:
            return HCurl
        if self.element.family in [EF.Regge]:
            return HEin
        if self.element.family in [EF.HHJ]:
            return HDivDiv

        return L2

    def tabulate(
        self, nderivs: int, points: _nda_f64
    ) -> _nda_f64:
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

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_BasixElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        For example, for a MixedElement, this will return the
        sub-element that represents the given component, the offset of
        that sub-element, and a stride of 1. For a BlockedElement, this
        will return the sub-element, an offset equal to the component
        number, and a stride equal to the block size. For vector-valued
        element (eg H(curl) and H(div) elements), this returns a
        ComponentElement (and as offset of 0 and a stride of 1). When
        tabulate is called on the ComponentElement, only the part of the
        table for the given component is returned.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """
        assert flat_component < self.value_size
        return ComponentElement(self, flat_component), 0, 1

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
    def reference_geometry(self) -> _nda_f64:
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
    def _wcoeffs(self) -> _nda_f64:
        """The coefficients used to define the polynomial set."""
        return self.element.wcoeffs

    @property
    def _x(self) -> _typing.List[_typing.List[_nda_f64]]:
        """The points used to define interpolation."""
        return self.element.x

    @property
    def _M(self) -> _typing.List[_typing.List[_nda_f64]]:
        """The matrices used to define interpolation."""
        return self.element.M

    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be computed
        as a tensor product of the basis elements of the elements in the factoriaation.
        """
        return self.element.has_tensor_product_factorisation

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        if not self.has_tensor_product_factorisation:
            return None
        return self.element.get_tensor_product_representation()


class ComponentElement(_BasixElementBase):
    """An element representing one component of a BasixElement."""

    element: _BasixElementBase
    component: int

    def __init__(self, element: _BasixElementBase, component: int):
        """Initialise the element."""
        self.element = element
        self.component = component
        super().__init__(
            f"ComponentElement({element._repr}, {component})", f"Component of {element.family_name}",
            element.cell_type.name, (1, ), element._degree)

    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return self.element.sobolev_space()

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (
            isinstance(other, ComponentElement) and self.element == other.element
            and self.component == other.component)

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(
        self, nderivs: int, points: _nda_f64
    ) -> _nda_f64:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions
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
                if isinstance(self.element, BlockedElement) and self.element.symmetric:
                    # FIXME: check that this behaves as expected
                    output.append(tbl[:, self.component, :])
                else:
                    vs0 = self.element._value_shape[0]
                    output.append(tbl[:, self.component // vs0, self.component % vs0, :])
            else:
                raise NotImplementedError()
        return _numpy.asarray(output, dtype=_numpy.float64)

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_BasixElementBase, int, int]:
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
    def reference_geometry(self) -> _nda_f64:
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
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return self.element.discontinuous

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return self.element.interpolation_nderivs


class MixedElement(_BasixElementBase):
    """A mixed element that combines two or more elements."""

    _sub_elements: _typing.List[_BasixElementBase]

    def __init__(self, sub_elements: _typing.List[_BasixElementBase]):
        """Initialise the element."""
        assert len(sub_elements) > 0
        self._sub_elements = sub_elements
        super().__init__(
            "MixedElement(" + ", ".join(i._repr for i in sub_elements) + ")",
            "mixed element", sub_elements[0].cell_type.name,
            (sum(i.value_size for i in sub_elements), ))

    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return max(e.sobolev_space() for e in self._sub_elements)

    def sub_elements(self) -> _typing.List[_BasixElementBase]:
        """Return a list of sub elements."""
        return self._sub_elements

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        if isinstance(other, MixedElement) and len(self._sub_elements) == len(other._sub_elements):
            for i, j in zip(self._sub_elements, other._sub_elements):
                if i != j:
                    return False
            return True
        return False

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    def tabulate(
        self, nderivs: int, points: _nda_f64
    ) -> _nda_f64:
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
            new_table = _numpy.zeros((len(points), self.value_size * self.dim))
            start = 0
            for e, t in zip(self._sub_elements, deriv_tables):
                for i in range(0, e.dim, e.value_size):
                    new_table[:, start: start + e.value_size] = t[:, i: i + e.value_size]
                    start += self.value_size
            tables.append(new_table)
        return _numpy.asarray(tables, dtype=_numpy.float64)

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_BasixElementBase, int, int]:
        """Get element that represents a component of the element, and the offset and stride of the component.

        Args:
            flat_component: The component

        Returns:
            component element, offset of the component, stride of the component
        """
        sub_dims = [0] + [e.dim for e in self._sub_elements]
        sub_cmps = [0] + [e.value_size for e in self._sub_elements]

        irange = _numpy.cumsum(sub_dims)
        crange = _numpy.cumsum(sub_cmps)

        # Find index of sub element which corresponds to the current
        # flat component
        component_element_index = _numpy.where(
            crange <= flat_component)[0].shape[0] - 1

        sub_e = self._sub_elements[component_element_index]

        e, offset, stride = sub_e.get_component_element(flat_component - crange[component_element_index])
        # TODO: is this offset correct?
        return e, irange[component_element_index] + offset, stride

    @property
    def ufcx_element_type(self) -> str:
        """Get the element type."""
        return "ufcx_mixed_element"

    @property
    def dim(self) -> int:
        """Number of DOFs the element has."""
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
        """Get the number of global support DOFs."""
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
    def reference_geometry(self) -> _nda_f64:
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
        """The number of derivatives needed when interpolating."""
        return max([e.interpolation_nderivs for e in self._sub_elements])


class BlockedElement(_BasixElementBase):
    """An element with a block size that contains multiple copies of a sub element."""

    block_shape: _typing.Tuple[int, ...]
    sub_element: _BasixElementBase
    _block_size: int

    def __init__(self, repr: str, sub_element: _BasixElementBase, block_size: int,
                 block_shape: _typing.Tuple[int, ...] = None, symmetric: bool = False):
        """Initialise the element."""
        assert block_size > 0
        if sub_element.value_size != 1:
            raise ValueError("Blocked elements (VectorElement and TensorElement) of "
                             "non-scalar elements are not supported. Try using MixedElement "
                             "instead.")

        self.sub_element = sub_element
        self._block_size = block_size
        if block_shape is None:
            block_shape = (block_size, )
        self.block_shape = block_shape
        self.symmetric = symmetric

        super().__init__(
            repr, sub_element.family(), sub_element.cell_type.name, block_shape,
            sub_element._degree, sub_element._map)

    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return self.sub_element.sobolev_space()

    def sub_elements(self) -> _typing.List[_BasixElementBase]:
        """Return a list of sub elements."""
        return [self.sub_element for _ in range(self._block_size)]

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return (
            isinstance(other, BlockedElement) and self._block_size == other._block_size
            and self.block_shape == other.block_shape and self.sub_element == other.sub_element)

    def __hash__(self) -> int:
        """Return a hash."""
        return super().__hash__()

    @property
    def block_size(self) -> int:
        """The block size of the element."""
        return self._block_size

    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Reference value shape of the element basis function."""
        if self.symmetric:
            assert len(self.block_shape) == 2 and self.block_shape[0] == self.block_shape[1]
            return (self.block_shape[0] * (self.block_shape[0] + 1) // 2, )
        return self._value_shape

    def tabulate(
        self, nderivs: int, points: _nda_f64
    ) -> _nda_f64:
        """Tabulate the basis functions of the element.

        Args:
            nderivs: Number of derivatives to tabulate.
            points: Points to tabulate at

        Returns:
            Tabulated basis functions
        """
        assert self.value_size == self._block_size  # TODO: remove this assumption
        output = []
        bs2 = self._block_size**2
        for table in self.sub_element.tabulate(nderivs, points):
            new_table = _numpy.zeros((table.shape[0], table.shape[1] * bs2))
            for block in range(self._block_size):
                col = block * (self._block_size + 1)
                new_table[:, col: col + table.shape[1] * bs2: bs2] = table
            output.append(new_table)
        return _numpy.asarray(output, dtype=_numpy.float64)

    def get_component_element(self, flat_component: int) -> _typing.Tuple[_BasixElementBase, int, int]:
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
    def reference_geometry(self) -> _nda_f64:
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
    def _wcoeffs(self) -> _nda_f64:
        """The coefficients used to define the polynomial set."""
        sub_wc = self.sub_element._wcoeffs
        wcoeffs = _numpy.zeros((sub_wc.shape[0] * self._block_size, sub_wc.shape[1] * self.block_size))
        for i in range(self._block_size):
            wcoeffs[sub_wc.shape[0] * i: sub_wc.shape[0] * (i + 1),
                    sub_wc.shape[1] * i: sub_wc.shape[1] * (i + 1)] = sub_wc
        return wcoeffs

    @property
    def _x(self) -> _typing.List[_typing.List[_nda_f64]]:
        """The points used to define interpolation."""
        return self.sub_element._x

    @property
    def _M(self) -> _typing.List[_typing.List[_nda_f64]]:
        """The matrices used to define interpolation."""
        M = []
        for M_list in self.sub_element._M:
            M_row = []
            for mat in M_list:
                new_mat = _numpy.zeros((mat.shape[0] * self._block_size, mat.shape[1] * self._block_size,
                                        mat.shape[2], mat.shape[3]))
                for i in range(self._block_size):
                    new_mat[i * mat.shape[0]: (i + 1) * mat.shape[0],
                            i * mat.shape[1]: (i + 1) * mat.shape[1], :, :] = mat
                M_row.append(new_mat)
            M.append(M_row)
        return M

    def has_tensor_product_factorisation(self) -> bool:
        """Indicates whether or not this element has a tensor product factorisation.

        If this value is true, this element's basis functions can be computed
        as a tensor product of the basis elements of the elements in the factoriaation.
        """
        return self.sub_element.has_tensor_product_factorisation()

    def get_tensor_product_representation(self):
        """Get the element's tensor product factorisation."""
        if not self.has_tensor_product_factorisation:
            return None
        return self.sub_element.get_tensor_product_representation()


class VectorElement(BlockedElement):
    """A vector element."""

    def __init__(self, sub_element: _BasixElementBase, size: int = None):
        """Initialise the element."""
        if size is None:
            size = len(_basix.topology(sub_element.cell_type)) - 1
        super().__init__(f"VectorElement({sub_element._repr}, {size})", sub_element, size, (size, ))


class TensorElement(BlockedElement):
    """A tensor element."""

    def __init__(self, sub_element: _BasixElementBase, shape: _typing.Tuple[int, int] = None, symmetric: bool = False):
        """Initialise the element."""
        if shape is None:
            size = len(_basix.topology(sub_element.cell_type)) - 1
            shape = (size, size)
        if symmetric:
            assert shape[0] == shape[1]
            bs = shape[0] * (shape[0] + 1) // 2
        else:
            bs = shape[0] * shape[1]
        assert len(shape) == 2
        super().__init__(f"TensorElement({sub_element._repr}, {shape})", sub_element, bs, shape, symmetric=symmetric)


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


@_functools.lru_cache()
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


@_functools.lru_cache()
def create_vector_element(
    family: _typing.Union[_basix.ElementFamily, str], cell: _typing.Union[_basix.CellType, str],
    degree: int, lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
    dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset, discontinuous=False
) -> VectorElement:
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
    return VectorElement(e)


@_functools.lru_cache()
def create_tensor_element(
    family: _typing.Union[_basix.ElementFamily, str], cell: _typing.Union[_basix.CellType, str],
    degree: int, lagrange_variant: _basix.LagrangeVariant = _basix.LagrangeVariant.unset,
    dpc_variant: _basix.DPCVariant = _basix.DPCVariant.unset, discontinuous=False
) -> TensorElement:
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
    return TensorElement(e)


def create_enriched_element(
    elements: _typing.List[_FiniteElementBase], preserve_functions: bool = None, preserve_dofs: bool = None
) -> _FiniteElementBase:
    """Create an enriched element from a list of elements.

    Args:
        elements: The elements to be combined
        preserve_functions: If this is set to True, the enriched element's basis functions will be the
            same as the basis functions of the input elements. This is the default behaviour
        preserve_dofs: If this is set to True, the enriched elements's DOF functionals will be the same
            as the DOF functionals of the input elements
    """
    if preserve_dofs is None and preserve_functions is None:
        preserve_functions = True
    if preserve_dofs is None:
        preserve_dofs = False
    if preserve_functions is None:
        preserve_functions = False

    if len([i for i in [preserve_functions, preserve_dofs] if i]) != 1:
        raise ValueError("Exactly one thing can be preserved by an enriched element.")

    ct = elements[0].cell_type
    vshape = elements[0].value_shape()
    vsize = elements[0].value_size
    mt = elements[0].map_type
    hcd = min(e.highest_complete_degree for e in elements)
    hd = max(e.highest_degree for e in elements)
    discontinuous = True
    for e in elements:
        if not e.discontinuous:
            discontinuous = False
        if e.cell_type != ct:
            raise ValueError("Enriched elements on different cell types not supported.")
        if e.value_shape() != vshape or e.value_size != vsize:
            raise ValueError("Enriched elements on different value shapes not supported.")
        if e.map_type != mt:
            raise ValueError("Enriched elements on different map types not supported.")
    nderivs = max(e.interpolation_nderivs for e in elements)

    # Create a coefficients matrix for polynomials that span the union off all the input elements' polynomial sets
    dim = sum(e.dim for e in elements)
    wcoeffs = _numpy.zeros((dim, _basix.polynomials.dim(_basix.PolynomialType.legendre, ct, hd) * vsize))
    row = 0
    for e in elements:
        wcoeffs[row: row + e.dim, :] = _basix.polynomials.reshape_coefficients(
            _basix.PolynomialType.legendre, ct, e._wcoeffs, vsize, e.highest_degree, hd)
        row += e.dim

    x = []
    M = []
    if preserve_dofs:
        # Create x and M from the inputs elements' x and M so that the DOFs are defined in the same
        # way as the input elements
        for pts_lists in zip(*[e._x for e in elements]):
            x.append([_numpy.concatenate(pts) for pts in zip(*pts_lists)])
        for M_lists in zip(*[e._M for e in elements]):
            M_row = []
            for M_parts in zip(*M_lists):
                ndofs = sum(mat.shape[0] for mat in M_parts)
                npts = sum(mat.shape[2] for mat in M_parts)
                deriv_dim = max(mat.shape[3] for mat in M_parts)
                new_M = _numpy.zeros((ndofs, vsize, npts, deriv_dim))
                pt = 0
                dof = 0
                for i, mat in enumerate(M_parts):
                    new_M[dof: dof + mat.shape[0], :, pt: pt + mat.shape[2], :mat.shape[3]] = mat
                    dof += mat.shape[0]
                    pt += mat.shape[2]
                M_row.append(new_M)
            M.append(M_row)
    elif preserve_functions:
        # Create DOFs that define an element whose basis functions are the basis functions
        # of the input elements
        geometry = _basix.geometry(ct)
        topology = _basix.topology(ct)
        connectivity = _basix.cell.sub_entity_connectivity(ct)
        tdim = len(topology) - 1
        first = True
        for d, t in enumerate(topology):
            if first:
                # If no DOFs have been defined yet, copy the definition from the input element
                x.append([_numpy.concatenate(pts) for pts in zip(*[e._x[d] for e in elements])])
                M_row = []
                for M_parts in zip(*[e._M[d] for e in elements]):
                    ndofs = sum(mat.shape[0] for mat in M_parts)
                    npts = sum(mat.shape[2] for mat in M_parts)
                    deriv_dim = max(mat.shape[3] for mat in M_parts)
                    new_M = _numpy.zeros((ndofs, vsize, npts, deriv_dim))
                    pt = 0
                    dof = 0
                    for i, mat in enumerate(M_parts):
                        new_M[dof: dof + mat.shape[0], :, pt: pt + mat.shape[2], :mat.shape[3]] = mat
                        dof += mat.shape[0]
                        pt += mat.shape[2]
                    if dof > 0:
                        first = False
                    M_row.append(new_M)
                M.append(M_row)
            else:
                # Create DOFs for sub-entities of dimension d
                x_entry = []
                M_entry = []
                for i, t2 in enumerate(t):
                    ndofs = sum(len(e.entity_dofs[d][i]) for e in elements)
                    deriv_dim = max(e._M[d][i].shape[3] for e in elements)

                    if ndofs == 0:
                        x_entry.append(_numpy.zeros((0, tdim)))
                        M_entry.append(_numpy.zeros((0, vsize, 0, deriv_dim)))
                        continue

                    # Get information about the sub-entity
                    origin = t2[0]
                    if d == 1:
                        assert len(t2) == 2
                        e_ct = _basix.CellType.interval
                        axes = [t2[1]]
                    elif d == 2:
                        if len(t2) == 3:
                            e_ct = _basix.CellType.triangle
                            axes = t2[1:]
                        elif len(t2) == 4:
                            e_ct = _basix.CellType.quadrilateral
                            axes = t2[1:3]
                        else:
                            raise ValueError(f"Unknown number of points for {d}D cell: {len(t2)}")
                    elif d == 3:
                        if len(t2) == 4:
                            e_ct = _basix.CellType.tetrahedron
                            axes = t2[1:]
                        elif len(t2) == 5:
                            e_ct = _basix.CellType.pyramid
                            axes = [t2[1], t2[2], t2[4]]
                        elif len(t2) == 6:
                            e_ct = _basix.CellType.prism
                            axes = [t2[1], t2[2], t2[3]]
                        elif len(t2) == 8:
                            e_ct = _basix.CellType.hexahedron
                            axes = [t2[1], t2[2], t2[4]]
                        else:
                            raise ValueError(f"Unknown number of points for {d}D cell: {len(t2)}")
                    else:
                        raise ValueError(f"Unsupported dimension: {d}")

                    # Make quadrature rule and map points to the sub-entity
                    pts, wts = _basix.make_quadrature(e_ct, 2 * hd)
                    npts = len(pts)
                    npoly = wcoeffs.shape[0]

                    mapped_pts = _numpy.array([geometry[origin] for p in pts])
                    for pi, c in enumerate(axes):
                        for j, p in enumerate(pts[:, pi]):
                            mapped_pts[j] += p * (geometry[c] - geometry[t2[0]])

                    # Add quadrature points to points defining the DOFs for this sub-entity
                    x_entry.append(mapped_pts)

                    # Tabulate all the input elements at the quadrature points
                    tabulations = []
                    for e in elements:
                        if e.block_size > 1:
                            assert e.block_size == e.value_size
                            table = e.tabulate(0, mapped_pts)[0]
                            new_table = _numpy.zeros((npts, table.shape[1] // e.block_size, e.block_size))
                            for b in range(e.block_size):
                                new_table[:, :, b] = table[:, b::e.block_size]
                            tabulations.append(new_table)
                        elif e.value_size > 1:
                            table = e.tabulate(0, mapped_pts)[0]
                            tdofs = table.shape[1] // e.value_size
                            new_table = _numpy.zeros((npts, tdofs, e.value_size))
                            for b in range(e.value_size):
                                new_table[:, :, b] = table[:, b * tdofs: (b + 1) * tdofs]
                            tabulations.append(new_table)
                        else:
                            tabulations.append(e.tabulate(0, mapped_pts)[0].reshape((npts, -1, e.value_size)))

                    # Compute orthogonal polynomials on the sub-entity that are included in the polynomial set
                    # entity_ortho_tab will be the evaluations of these orthogonal polynomials at the quadrature
                    # points
                    otab = _basix.tabulate_polynomials(_basix.PolynomialType.legendre, ct, hd, mapped_pts)
                    ortho_tab = _numpy.zeros((otab.shape[0] * vsize, npts * vsize))
                    for v in range(vsize):
                        ortho_tab[otab.shape[0] * v: otab.shape[0] * (v + 1), v * npts: (v + 1) * npts] = otab
                    ortho_tab = (wcoeffs @ ortho_tab).reshape((npoly, npts, vsize))

                    orthogonal_data: _typing.List[_nda_f64] = []
                    for table, e in zip(tabulations, elements):
                        for p in range(e.dim):
                            coeffs = _numpy.array([
                                sum(table[i, p, v] * ortho_tab[q, i, v] for i in range(npts) for v in range(vsize))
                                for q in range(npoly)])
                            for o in orthogonal_data:
                                coeffs -= _numpy.dot(o, coeffs) * o
                            if not _numpy.isclose(sum(abs(i) for i in coeffs), 0):
                                coeffs /= _numpy.dot(coeffs, coeffs) ** 0.5
                                orthogonal_data.append(coeffs)
                    orthogonal = _numpy.array(orthogonal_data)

                    entity_ortho_tab = (orthogonal @ ortho_tab.reshape(npoly, -1)).reshape((-1, npts, vsize))

                    # Evaluate the basis functions of each input element at the quadrature points
                    rows = _numpy.zeros(entity_ortho_tab.shape)
                    row_n = 0

                    entities = [(d, i)] + [(d2, entity) for d2 in range(d) for entity in connectivity[d][i][d2]]
                    for d2, entity in entities:
                        for e, tab in zip(elements, tabulations):
                            for dof in e.entity_dofs[d2][entity]:
                                rows[row_n, :, :] = tab[:, dof, :]
                                row_n += 1
                    assert row_n == rows.shape[0]

                    # Create matrix whose entries are the coefficients (in terms of the entity orthogonal polynomials)
                    # of tha basis functions of the input elements
                    matrix = _numpy.array([[sum(sum(i) for i in f * g) for g in entity_ortho_tab] for f in rows])

                    # Use this matrix to compute functions that can be used as integral moments
                    M_mat = _numpy.zeros((ndofs, vsize, npts, deriv_dim))
                    for dof in range(ndofs):
                        rhs = _numpy.array([1.0 if i == dof else 0.0 for i in range(matrix.shape[1])])
                        coeffs = _numpy.linalg.solve(matrix, rhs)
                        for v in range(vsize):
                            M_mat[dof, v, :, 0] = coeffs @ entity_ortho_tab[:, :, v]

                    M_entry.append(M_mat)

                x.append(x_entry)
                M.append(M_entry)

        # Include empty points and operators for dimensions larger than the tdim of the cell
        for i in range(tdim + 1, 4):
            x.append([])
            M.append([])

    return BasixElement(
        _basix.create_custom_element(ct, vshape, wcoeffs, x, M, nderivs, mt, discontinuous, hcd, hd),
        repr="EnrichedElement(" + ", ".join(repr(e) for e in elements) + ")")


def convert_ufl_element(
    element: _FiniteElementBase
) -> _BasixElementBase:
    """Convert a UFL element to a wrapped Basix element."""
    if isinstance(element, _BasixElementBase):
        return element

    elif isinstance(element, _ufl.VectorElement):
        return VectorElement(convert_ufl_element(element.sub_elements()[0]), element.num_sub_elements())
    elif isinstance(element, _ufl.TensorElement):
        return TensorElement(convert_ufl_element(element.sub_elements()[0]), element._value_shape)
    elif isinstance(element, _ufl.MixedElement):
        return MixedElement([convert_ufl_element(e) for e in element.sub_elements()])
    elif isinstance(element, _ufl.EnrichedElement):
        return create_enriched_element([convert_ufl_element(e) for e in element._elements])

    # Elements that will not be supported
    elif isinstance(element, _ufl.NodalEnrichedElement):
        raise RuntimeError("NodalEnrichedElement is not supported. Use EnrichedElement instead.")
    elif isinstance(element, _ufl.BrokenElement):
        raise RuntimeError("BrokenElement not supported.")
    elif isinstance(element, _ufl.HCurlElement):
        raise RuntimeError("HCurlElement not supported.")
    elif isinstance(element, _ufl.HDivElement):
        raise RuntimeError("HDivElement not supported.")
    elif isinstance(element, _ufl.TensorProductElement):
        raise RuntimeError("TensorProductElement not supported.")
    elif isinstance(element, _ufl.RestrictedElement):
        raise RuntimeError("RestricedElement not supported.")

    elif isinstance(element, _ufl.FiniteElement):
        # Create a basix element from family name, cell type and degree
        family_name = element.family()
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

        family_type = _basix.finite_element.string_to_family(family_name, element.cell().cellname())
        cell_type = _basix.cell.string_to_type(element.cell().cellname())

        variant_info = {
            "lagrange_variant": _basix.LagrangeVariant.unset,
            "dpc_variant": _basix.DPCVariant.unset
        }
        if family_type == _basix.ElementFamily.P and element.variant() == "equispaced":
            # This is used for elements defining cells
            variant_info["lagrange_variant"] = _basix.LagrangeVariant.equispaced
        else:
            if element.variant() is not None:
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

        return create_element(family_type, cell_type, element.degree(), **variant_info, discontinuous=discontinuous)

    else:
        raise ValueError(f"Unrecognised element type: {element.__class__.__name__}")
