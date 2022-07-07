"""Functions to directly wrap Basix elements in UFL."""

import ufl as _ufl
from ufl.finiteelement.finiteelementbase import FiniteElementBase as _FiniteElementBase
import hashlib as _hashlib
import numpy as _numpy
import basix as _basix
import typing as _typing
import functools as _functools

if _typing.TYPE_CHECKING:
    _nda_f64 = _numpy.typing.NDArray[_numpy.float64]
else:
    _nda_f64 = None


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

    def value_shape(self) -> _typing.Tuple[int, ...]:
        """Value shape of the element basis function."""
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


class BasixElement(_BasixElementBase):
    """A wrapper allowing Basix elements to be used directly with UFL."""

    element: _basix.finite_element.FiniteElement

    def __init__(self, element: _basix.finite_element.FiniteElement):
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
            _map_type_to_string(element.map_type))

        self.element = element

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        return isinstance(other, BasixElement) and self.element == other.element

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
                # TODO: Something different may need doing here if
                # tensor is symmetric
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

    sub_elements: _typing.List[_BasixElementBase]

    def __init__(self, sub_elements: _typing.List[_BasixElementBase]):
        """Initialise the element."""
        assert len(sub_elements) > 0
        vs: _typing.Tuple[int, ...] = tuple()
        for i in sub_elements:
            vs += i._value_shape
        self.sub_elements = sub_elements
        super().__init__(
            "MixedElement(" + ", ".join(i._repr for i in sub_elements) + ")",
            "mixed element", sub_elements[0].cell_type.name, vs)

    def __eq__(self, other) -> bool:
        """Check if two elements are equal."""
        if isinstance(other, MixedElement) and len(self.sub_elements) == len(other.sub_elements):
            for i, j in zip(self.sub_elements, other.sub_elements):
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
        results = [e.tabulate(nderivs, points) for e in self.sub_elements]
        for deriv_tables in zip(*results):
            new_table = _numpy.zeros((len(points), self.value_size * self.dim))
            start = 0
            for e, t in zip(self.sub_elements, deriv_tables):
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
        sub_dims = [0] + [e.dim for e in self.sub_elements]
        sub_cmps = [0] + [e.value_size for e in self.sub_elements]

        irange = _numpy.cumsum(sub_dims)
        crange = _numpy.cumsum(sub_cmps)

        # Find index of sub element which corresponds to the current
        # flat component
        component_element_index = _numpy.where(
            crange <= flat_component)[0].shape[0] - 1

        sub_e = self.sub_elements[component_element_index]

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
        return sum(e.dim for e in self.sub_elements)

    @property
    def num_entity_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with each entity."""
        data = [e.num_entity_dofs for e in self.sub_elements]
        return [[sum(d[tdim][entity_n] for d in data) for entity_n, _ in enumerate(entities)]
                for tdim, entities in enumerate(data[0])]

    @property
    def entity_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with each entity."""
        dofs: _typing.List[_typing.List[_typing.List[int]]] = [
            [[] for i in entities]
            for entities in self.sub_elements[0].entity_dofs]
        start_dof = 0
        for e in self.sub_elements:
            for tdim, entities in enumerate(e.entity_dofs):
                for entity_n, entity_dofs in enumerate(entities):
                    dofs[tdim][entity_n] += [start_dof + i for i in entity_dofs]
            start_dof += e.dim
        return dofs

    @property
    def num_entity_closure_dofs(self) -> _typing.List[_typing.List[int]]:
        """Number of DOFs associated with the closure of each entity."""
        data = [e.num_entity_closure_dofs for e in self.sub_elements]
        return [[sum(d[tdim][entity_n] for d in data) for entity_n, _ in enumerate(entities)]
                for tdim, entities in enumerate(data[0])]

    @property
    def entity_closure_dofs(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """DOF numbers associated with the closure of each entity."""
        dofs: _typing.List[_typing.List[_typing.List[int]]] = [
            [[] for i in entities]
            for entities in self.sub_elements[0].entity_closure_dofs]
        start_dof = 0
        for e in self.sub_elements:
            for tdim, entities in enumerate(e.entity_closure_dofs):
                for entity_n, entity_dofs in enumerate(entities):
                    dofs[tdim][entity_n] += [start_dof + i for i in entity_dofs]
            start_dof += e.dim
        return dofs

    @property
    def num_global_support_dofs(self) -> int:
        """Get the number of global support DOFs."""
        return sum(e.num_global_support_dofs for e in self.sub_elements)

    @property
    def family_name(self) -> str:
        """Family name of the element."""
        return "mixed element"

    @property
    def reference_topology(self) -> _typing.List[_typing.List[_typing.List[int]]]:
        """Topology of the reference element."""
        return self.sub_elements[0].reference_topology

    @property
    def reference_geometry(self) -> _nda_f64:
        """Geometry of the reference element."""
        return self.sub_elements[0].reference_geometry

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
        return self.sub_elements[0].cell_type

    @property
    def discontinuous(self) -> bool:
        """True if the discontinuous version of the element is used."""
        return False

    @property
    def interpolation_nderivs(self) -> int:
        """The number of derivatives needed when interpolating."""
        return max([e.interpolation_nderivs for e in self.sub_elements])


class BlockedElement(_BasixElementBase):
    """An element with a block size that contains multiple copies of a sub element."""

    block_shape: _typing.Tuple[int, ...]
    sub_element: _BasixElementBase
    _block_size: int

    def __init__(self, repr: str, sub_element: _BasixElementBase, block_size: int,
                 block_shape: _typing.Tuple[int, ...] = None):
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

        super().__init__(
            repr, "blocked element", sub_element.cell_type.name, block_shape,
            sub_element._degree, sub_element._map)

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
        assert len(self.block_shape) == 1  # TODO: block shape
        assert self.value_size == self._block_size  # TODO: remove this assumption
        output = []
        for table in self.sub_element.tabulate(nderivs, points):
            new_table = _numpy.zeros((table.shape[0], table.shape[1] * self._block_size**2))
            for block in range(self._block_size):
                col = block * (self._block_size + 1)
                new_table[:, col: col + table.shape[1] * self._block_size**2: self._block_size**2] = table
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


class VectorElement(BlockedElement):
    """A vector element."""

    def __init__(self, sub_element: _BasixElementBase, size: int = None):
        """Initialise the element."""
        if size is None:
            size = len(_basix.topology(sub_element.cell_type)) - 1
        super().__init__(f"VectorElement({sub_element._repr}, {size})", sub_element, size, (size, ))


class TensorElement(BlockedElement):
    """A tensor element."""

    def __init__(self, sub_element: _BasixElementBase, shape: _typing.Tuple[int, int] = None):
        """Initialise the element."""
        if shape is None:
            size = len(_basix.topology(sub_element.cell_type)) - 1
            shape = (size, size)
        assert len(shape) == 2
        super().__init__(f"TensorElement({sub_element._repr}, {shape})", sub_element, shape[0] * shape[1], shape)


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

    # Elements not yet supported
    elif isinstance(element, _ufl.EnrichedElement):
        raise NotImplementedError()

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
