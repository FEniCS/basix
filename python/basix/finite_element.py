# Copyright (C) 2023-2024 Matthew Scroggs
#
# This file is part of Basix (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT
"""Functions for creating finite elements."""

import typing

import numpy as np
import numpy.typing as npt

from basix._basixcpp import DPCVariant as _DPCV
from basix._basixcpp import ElementFamily as _EF
from basix._basixcpp import FiniteElement_float32 as _FiniteElement_float32
from basix._basixcpp import FiniteElement_float64 as _FiniteElement_float64
from basix._basixcpp import LagrangeVariant as _LV
from basix._basixcpp import create_custom_element as _create_custom_element
from basix._basixcpp import create_element as _create_element
from basix._basixcpp import create_tp_element as _create_tp_element
from basix._basixcpp import tp_dof_ordering as _tp_dof_ordering
from basix._basixcpp import tp_factors as _tp_factors
from basix.cell import CellType
from basix.maps import MapType
from basix.polynomials import PolysetType
from basix.sobolev_spaces import SobolevSpace
from basix.utils import Enum

__all__ = [
    "FiniteElement",
    "create_element",
    "create_custom_element",
    "create_tp_element",
    "string_to_family",
    "string_to_lagrange_variant",
    "string_to_dpc_variant",
    "tp_factors",
    "tp_dof_ordering",
]


class ElementFamily(Enum):
    """Element family."""

    custom = _EF.custom
    P = _EF.P
    BDM = _EF.BDM
    RT = _EF.RT
    N1E = _EF.N1E
    N2E = _EF.N2E
    Regge = _EF.Regge
    HHJ = _EF.HHJ
    bubble = _EF.bubble
    serendipity = _EF.serendipity
    DPC = _EF.DPC
    CR = _EF.CR
    Hermite = _EF.Hermite
    iso = _EF.iso


class LagrangeVariant(Enum):
    """Lagrange variant."""

    unset = _LV.unset
    equispaced = _LV.equispaced
    gll_warped = _LV.gll_warped
    gll_isaac = _LV.gll_isaac
    gll_centroid = _LV.gll_centroid
    chebyshev_warped = _LV.chebyshev_warped
    chebyshev_isaac = _LV.chebyshev_isaac
    chebyshev_centroid = _LV.chebyshev_centroid
    gl_warped = _LV.gl_warped
    gl_isaac = _LV.gl_isaac
    gl_centroid = _LV.gl_centroid
    legendre = _LV.legendre
    bernstein = _LV.bernstein


class DPCVariant(Enum):
    """DPC variant."""

    unset = _DPCV.unset
    simplex_equispaced = _DPCV.simplex_equispaced
    simplex_gll = _DPCV.simplex_gll
    horizontal_equispaced = _DPCV.horizontal_equispaced
    horizontal_gll = _DPCV.horizontal_gll
    diagonal_equispaced = _DPCV.diagonal_equispaced
    diagonal_gll = _DPCV.diagonal_gll
    legendre = _DPCV.legendre


class FiniteElement:
    """Finite element class."""

    _e: typing.Union[_FiniteElement_float32, _FiniteElement_float64]

    def __init__(self, e: typing.Union[_FiniteElement_float32, _FiniteElement_float64]):
        """Initialise a finite element wrapper.

        Note:
            This initialiser is intended for internal library use.
        """
        self._e = e

    def tabulate(self, n: int, x: npt.NDArray) -> npt.NDArray[np.floating]:
        """Compute basis values and derivatives at set of points.

        Note:
            The version of `FiniteElement::tabulate` with the basis data
            as an out argument should be preferred for repeated call
            where performance is critical

        Args:
            n: The order of derivatives, up to and including, to
                compute. Use 0 for the basis functions only.
            x: The points at which to compute the basis functions. The
                shape of x is (number of points, geometric dimension).

        Returns:
            The basis functions (and derivatives). The shape is
            ``(derivative, point, basis fn index, value index)``.

            * The first index is the derivative, with higher derivatives
                are stored in triangular (2D) or tetrahedral (3D)
                ordering, i.e. for the ``(x,y)`` derivatives in 2D:
                ``(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0)...``.
                The function basix::indexing::idx can be used to find
                the appropriate derivative.

            * The second index is the point index

            * The fourth index is the basis function component. Its has size

            * The third index is the basis function index one for scalar
                basis functions.
        """
        return self._e.tabulate(n, x)

    def __eq__(self, other) -> bool:
        """Test element for equality."""
        try:
            return self._e == other._e
        except TypeError:
            return False

    def __hash__(self) -> int:
        """Hash."""
        return hash(self._e)

    def push_forward(self, U, J, detJ, K) -> npt.NDArray[np.floating]:
        """Map function values from the reference to a physical cell.

        This function can perform the mapping for multiple points,
        grouped by points that share a common Jacobian.

        Args:
            U: The function values on the reference cell. The indices are
                ``(Jacobian index, point index, components)``.
            J: The Jacobian of the mapping. The indices are ``(Jacobian
                index, J_i, J_j)``.
            detJ: The determinant of the Jacobian of the mapping. It has
                length ``J.shape(0)``.
            K: The inverse of the Jacobian of the
               mapping. The indices are ``(Jacobian index, K_i, K_j)``.

        Returns:
            The function values on the cell. The indices are ``(Jacobian
            index, point index, components)``.
        """
        return self._e.push_forward(U, J, detJ, K)

    def pull_back(
        self, u: npt.NDArray, J: npt.NDArray, detJ: npt.NDArray, K: npt.NDArray
    ) -> npt.NDArray[np.floating]:
        """Map function values from a physical cell to the reference.

        Args:
            u: The function values on the cell.
            J: The Jacobian of the mapping.
            detJ: The determinant of the Jacobian of the mapping.
            K: The inverse of the Jacobian of the mapping.

        Returns:
            The function values on the reference. The indices are
            ``(Jacobian index, point index, components``).
        """
        return self._e.pull_back(u, J, detJ, K)

    def T_apply(self, data, block_size, cell_info) -> None:
        """Apply DOF transformations to some data in-place.

        Note:
            This function is designed to be called at runtime, so its
            performance is critical.

        Args:
            data: The data
            block_size: The number of data points per DOF
            cell_info: The permutation info for the cell

        """
        self._e.T_apply(data, block_size, cell_info)

    def Tt_apply_right(self, data, block_size, cell_info) -> None:
        """Post-apply DOF transformations to some transposed data in-place.

        Note:
            This function is designed to be called at runtime, so its
            performance is critical.

        Args:
            data: The data.
            block_size: The number of data points per DOF.
            cell_info: The permutation info for the cell.
        """
        self._e.Tt_apply_right(data, block_size, cell_info)

    def Tt_inv_apply(self, data, block_size, cell_info) -> None:
        """Pre-apply inverse transpose DOF transformations to some data.

        Note:
            This function is designed to be called at runtime, so its
            performance is critical.

        Args:
            data: The data.
            block_size: The number of data points per DOF.
            cell_info: The permutation info for the cell.
        """
        self._e.Tt_inv_apply(data, block_size, cell_info)

    def base_transformations(self) -> npt.NDArray[np.floating]:
        r"""Get the base transformations.

        The base transformations represent the effect of rotating or
        reflecting a subentity of the cell on the numbering and
        orientation of the DOFs. This returns a list of matrices with
        one matrix for each subentity permutation in the following
        order:
        Reversing edge 0, reversing edge 1, ...
        Rotate face 0, reflect face 0, rotate face 1, reflect face 1, ...

        *Example: Order 3 Lagrange on a triangle*

        This space has 10 dofs arranged like:

        .. code-block::

            2
            |\
            6 4
            |  \
            5 9 3
            |    \
            0-7-8-1

        For this element, the base transformations are:
        [Matrix swapping 3 and 4,
        Matrix swapping 5 and 6,
        Matrix swapping 7 and 8]
        The first row shows the effect of reversing the diagonal edge. The
        second row shows the effect of reversing the vertical edge. The third
        row shows the effect of reversing the horizontal edge.

        *Example: Order 1 Raviart-Thomas on a triangle*

        This space has 3 dofs arranged like:

        .. code-block::

              |\
              | \
              |  \
            <-1   0
              |  / \
              | L ^ \
              |   |  \
               ---2---

        These DOFs are integrals of normal components over the edges: DOFs 0 and 2
        are oriented inward, DOF 1 is oriented outwards.
        For this element, the base transformation matrices are:

        .. code-block::

            0: [[-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 1]]
            1: [[1,  0, 0],
                [0, -1, 0],
                [0,  0, 1]]
            2: [[1, 0,  0],
                [0, 1,  0],
                [0, 0, -1]]


        The first matrix reverses DOF 0 (as this is on the first edge). The second
        matrix reverses DOF 1 (as this is on the second edge). The third matrix
        reverses DOF 2 (as this is on the third edge).

        *Example: DOFs on the face of Order 2 Nedelec first kind on a tetrahedron*

        On a face of this tetrahedron, this space has two face tangent DOFs:

        .. code-block::

            |\        |\
            | \       | \
            |  \      | ^\
            |   \     | | \
            | 0->\    | 1  \
            |     \   |     \
             ------    ------

        For these DOFs, the subblocks of the base transformation matrices are:

        .. code-block::

            rotation: [[-1, 1],
                       [ 1, 0]]
            reflection: [[0, 1],
                         [1, 0]]

        Returns:
            The base transformations for this element. The shape is
            ``(ntranformations, ndofs, ndofs)``.
        """
        return self._e.base_transformations()

    def entity_transformations(self) -> dict:
        """Entity dof transformation matrices.

        Returns:
            The base transformations for this element. The shape is
            ``(ntranformations, ndofs, ndofs)``.
        """
        return self._e.entity_transformations()

    def get_tensor_product_representation(self) -> list[list["FiniteElement"]]:
        """Get the tensor product representation of this element.

        Raises an exception if no such factorisation exists.

        The tensor product representation will be a vector of tuples.
        Each tuple contains a vector of finite elements, and a vector of
        integers. The vector of finite elements gives the elements on an
        interval that appear in the tensor product representation. The
        vector of integers gives the permutation between the numbering
        of the tensor product DOFs and the number of the DOFs of this
        Basix element.

        Returns:
            The tensor product representation
        """
        factors = self._e.get_tensor_product_representation()
        return [[FiniteElement(e) for e in elements] for elements in factors]

    @property
    def degree(self) -> int:
        """Element polynomial degree."""
        return self._e.degree

    @property
    def embedded_superdegree(self) -> int:
        """Embedded polynomial degree.

        Lowest degree ``n`` such that the highest degree polynomial in
        this element is contained in a Lagrange (or vector Lagrange)
        element of degree ``n``.
        """
        return self._e.embedded_superdegree

    @property
    def embedded_subdegree(self) -> int:
        """Embedded polynomial sub-degree.

        Highest degree ``n`` such that a Lagrange (or vector Lagrange)
        element of degree ``n`` is a subspace of this element.
        """
        return self._e.embedded_subdegree

    @property
    def cell_type(self) -> CellType:
        """Element cell type."""
        return getattr(CellType, self._e.cell_type.name)

    @property
    def polyset_type(self) -> PolysetType:
        """Element polyset type."""
        return getattr(PolysetType, self._e.polyset_type.name)

    @property
    def dim(self) -> int:
        """Dimension of the finite element space.

        This is the number of degrees-of-freedom for the element.
        """
        return self._e.dim

    @property
    def num_entity_dofs(self) -> list[list[int]]:
        """Number of entity dofs.

        Warning:
            This property may be removed.
        """
        return self._e.num_entity_dofs

    @property
    def entity_dofs(self) -> list[list[list[int]]]:
        """Dofs on each topological entity.

        Data is order ``(vertices, edges, faces, cell)``. For example,
        Lagrange degree 2 on a triangle has vertices: ``[[0], [1],
        [2]]``, edges: ``[[3], [4], [5]]``, cell: ``[[]]``.
        """
        return self._e.entity_dofs

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]:
        """Number of entity closure dofs.

        Warning:
            This property may be removed.
        """
        return self._e.num_entity_closure_dofs

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]:
        """Get the dofs on the closure of each topological entity.

        Data is in the order ``(vertices, edges, faces, cell)``. For
        example, Lagrange degree 2 on a triangle has vertices: ``[[0],
        [1], [2]]``, edges: ``[[1, 2, 3], [0, 2, 4], [0, 1, 5]]``, cell:
        ``[[0, 1, 2, 3, 4, 5]]``.
        """
        return self._e.entity_closure_dofs

    @property
    def value_size(self) -> int:
        """Value size."""
        return self._e.value_size

    @property
    def value_shape(self) -> list[int]:
        """Element value tensor shape.

        E.g., returning ``(,)`` for scalars, ``(3,)`` for vectors in 3D,
        ``(2, 2)`` for a rank-2 tensor in 2D.
        """
        return self._e.value_shape

    @property
    def discontinuous(self) -> bool:
        """True is element is the discontinuous variant."""
        return self._e.discontinuous

    @property
    def family(self) -> ElementFamily:
        """Finite element family."""
        return getattr(ElementFamily, self._e.family.name)

    @property
    def lagrange_variant(self) -> LagrangeVariant:
        """Lagrange variant of the element."""
        return getattr(LagrangeVariant, self._e.lagrange_variant.name)

    @property
    def dpc_variant(self) -> DPCVariant:
        """DPC variant of the element."""
        return getattr(DPCVariant, self._e.dpc_variant.name)

    @property
    def dof_transformations_are_permutations(self) -> bool:
        """True if the dof transformations are all permutations."""
        return self._e.dof_transformations_are_permutations

    @property
    def dof_transformations_are_identity(self) -> bool:
        """True if DOF transformations are all the identity."""
        return self._e.dof_transformations_are_identity

    @property
    def interpolation_is_identity(self) -> bool:
        """True if interpolation matrix for this element is the identity."""
        return self._e.interpolation_is_identity

    @property
    def map_type(self) -> MapType:
        """Map type for this element."""
        return getattr(MapType, self._e.map_type.name)

    @property
    def sobolev_space(self) -> SobolevSpace:
        """Underlying Sobolev space for this element."""
        return getattr(SobolevSpace, self._e.sobolev_space.name)

    @property
    def points(self) -> npt.NDArray[np.floating]:
        """Interpolation points.

        Coordinates on the reference element where a function need to be
        evaluated in order to interpolate it in the finite element
        space. Shape is ``(num_points, tdim)``.
        """
        return self._e.points

    @property
    def interpolation_matrix(self) -> npt.NDArray[np.floating]:
        """Interpolation points.

        Coordinates on the reference element where a function need to be
        evaluated in order to interpolate it in the finite element
        space.
        """
        return self._e.interpolation_matrix

    @property
    def dual_matrix(self) -> npt.NDArray[np.floating]:
        """Matrix $BD^{T}$.

        See C++ documentation.
        """
        return self._e.dual_matrix

    @property
    def coefficient_matrix(self) -> npt.NDArray[np.floating]:
        """Matrix of coefficients."""
        return self._e.coefficient_matrix

    @property
    def wcoeffs(self) -> npt.NDArray[np.floating]:
        """Coefficients that define the polynomial set in terms of the orthonormal polynomials.

        See C++ documentation for details.
        """
        return self._e.wcoeffs

    @property
    def M(self) -> list[list[npt.NDArray[np.floating]]]:
        """Interpolation matrices for each sub-entity.

        See C++ documentation for details.
        """
        return self._e.M

    @property
    def x(self) -> list[list[npt.NDArray[np.floating]]]:
        """Interpolation points for each sub-entity.

        The indices of this data are ``(tdim, entity index, point index,
        dim)``.
        """
        return self._e.x

    @property
    def has_tensor_product_factorisation(self) -> bool:
        """True if element has tensor-product structure."""
        return self._e.has_tensor_product_factorisation

    @property
    def interpolation_nderivs(self) -> int:
        """Number of derivatives needed when interpolating."""
        return self._e.interpolation_nderivs

    @property
    def dof_ordering(self) -> list[int]:
        """DOF layout."""
        return self._e.dof_ordering

    @property
    def dtype(self) -> npt.DTypeLike:
        """Element float type."""
        return np.dtype(self._e.dtype)


def create_element(
    family: ElementFamily,
    celltype: CellType,
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.unset,
    dpc_variant: DPCVariant = DPCVariant.unset,
    discontinuous: bool = False,
    dof_ordering: typing.Optional[list[int]] = None,
    dtype: npt.DTypeLike = np.float64,
) -> FiniteElement:
    """Create a finite element.

    Args:
        family: Finite element family.
        celltype: Reference cell type that the element is defined on.
        degree: Polynomial degree of the element.
        lagrange_variant: Lagrange variant type.
        dpc_variant: DPC variant type.
        discontinuous: If `True` element is discontinuous. The
            discontinuous element will have the same DOFs as a
            continuous element, but the DOFs will all be associated with
            the interior of the cell.
        dof_ordering: Ordering of dofs for ``ElementDofLayout``.
        dtype: Element scalar type.

    Returns:
        A finite element.
    """
    return FiniteElement(
        _create_element(
            family.value,
            celltype.value,
            degree,
            lagrange_variant.value,
            dpc_variant.value,
            discontinuous,
            dof_ordering if dof_ordering is not None else [],
            np.dtype(dtype).char,
        )
    )


def create_custom_element(
    cell_type: CellType,
    value_shape: tuple[int, ...],
    wcoeffs: npt.NDArray[np.floating],
    x: list[list[npt.NDArray[np.floating]]],
    M: list[list[npt.NDArray[np.floating]]],
    interpolation_nderivs: int,
    map_type: MapType,
    sobolev_space: SobolevSpace,
    discontinuous: bool,
    embedded_subdegree: int,
    embedded_superdegree: int,
    poly_type: PolysetType,
) -> FiniteElement:
    """Create a custom finite element.

    Args:
        cell_type: Element cell type.
        value_shape: Value shape of the element.
        wcoeffs: Matrices for the k-th value index containing the
            expansion coefficients defining a polynomial basis spanning
            the polynomial space for this element. Shape is
            ``(dim(finite element polyset), dim(Legendre
            polynomials))``.
        x: Interpolation points. Indices are ``(tdim, entity index,
            point index, dim)``.
        M: Interpolation matrices. Indices are ``(tdim, entity
            index, dof, vs, point_index, derivative)``.
        interpolation_nderivs: Number of derivatives that need to be
            used during interpolation.
        map_type: Type of map to be used to map values from the
            reference to a physical cell.
        sobolev_space: Underlying Sobolev space for the element.
        discontinuous: If ``True`` create the discontinuous version of
            the element.
        embedded_subdegree: Highest degree n such that a Lagrange
            (or vector Lagrange) element of degree n is a subspace of
            this element.
        embedded_superdegree: Degree of a polynomial in this
            element's polyset.
        poly_type: Type of polyset to use for this element.

    Returns:
        A custom finite element.
    """
    return FiniteElement(
        _create_custom_element(
            cell_type.value,
            value_shape,
            wcoeffs,
            x,
            M,
            interpolation_nderivs,
            map_type.value,
            sobolev_space.value,
            discontinuous,
            embedded_subdegree,
            embedded_superdegree,
            poly_type.value,
        )
    )


def create_tp_element(
    family: ElementFamily,
    celltype: CellType,
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.unset,
    dpc_variant: DPCVariant = DPCVariant.unset,
    discontinuous: bool = False,
    dtype: npt.DTypeLike = np.float64,
) -> FiniteElement:
    """Create a finite element with tensor product ordering.

    Args:
        family: Finite element family.
        celltype: Reference cell type that the element is defined on
        degree: Polynomial degree of the element.
        lagrange_variant: Lagrange variant type.
        dpc_variant: DPC variant type.
        discontinuous: If `True` element is discontinuous. The
            discontinuous element will have the same DOFs as a
            continuous element, but the DOFs will all be associated with
            the interior of the cell.
        dtype: Element scalar type.

    Returns:
        A finite element.
    """
    return FiniteElement(
        _create_tp_element(
            family.value,
            celltype.value,
            degree,
            lagrange_variant.value,
            dpc_variant.value,
            discontinuous,
            np.dtype(dtype).char,
        )
    )


def tp_factors(
    family: ElementFamily,
    celltype: CellType,
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.unset,
    dpc_variant: DPCVariant = DPCVariant.unset,
    discontinuous: bool = False,
    dof_ordering: typing.Optional[list[int]] = None,
    dtype: npt.DTypeLike = np.float64,
) -> list[list[FiniteElement]]:
    """Elements in the tensor product factorisation of an element.

    If the element has no factorisation, an empty list is returned.

    Args:
        family: Finite element family.
        celltype: Reference cell type that the element is defined on
        degree: Polynomial degree of the element.
        lagrange_variant: Lagrange variant type.
        dpc_variant: DPC variant type.
        discontinuous: If `True` element is discontinuous. The
            discontinuous element will have the same DOFs as a
            continuous element, but the DOFs will all be associated with
            the interior of the cell.
        dof_ordering: Ordering of dofs for ElementDofLayout
        dtype: Element scalar type.

    Returns:
        A list of finite elements.
    """
    return [
        [FiniteElement(e) for e in elements]
        for elements in _tp_factors(
            family.value,
            celltype.value,
            degree,
            lagrange_variant.value,
            dpc_variant.value,
            discontinuous,
            dof_ordering if dof_ordering is not None else [],
            np.dtype(dtype).char,
        )
    ]


def tp_dof_ordering(
    family: ElementFamily,
    celltype: CellType,
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.unset,
    dpc_variant: DPCVariant = DPCVariant.unset,
    discontinuous: bool = False,
) -> list[int]:
    """Tensor product DOF ordering for an element.

    This DOF ordering can be passed into create_element to create the
    element with DOFs ordered in a tensor product order.

    If the element has no tensor product factorisation, an empty list is
    returned.

    Args:
        family: Finite element family.
        celltype: Reference cell type that the element is defined on
        degree: Polynomial degree of the element.
        lagrange_variant: Lagrange variant type.
        dpc_variant: DPC variant type.
        discontinuous: If `True` element is discontinuous. The
            discontinuous element will have the same DOFs as a
            continuous element, but the DOFs will all be associated with
            the interior of the cell.

    Returns:
        The DOF ordering.
    """
    return _tp_dof_ordering(
        family.value,
        celltype.value,
        degree,
        lagrange_variant.value,
        dpc_variant.value,
        discontinuous,
    )


def string_to_family(family: str, cell: str) -> ElementFamily:
    """Basix ElementFamily enum representing the family type on the given cell.

    Args:
        family: Element family as a string.
        cell: Cell type as a string.

    Returns:
        Element family.
    """
    # Family names that are valid for all cells
    families = {
        "Lagrange": ElementFamily.P,
        "P": ElementFamily.P,
        "Bubble": ElementFamily.bubble,
        "bubble": ElementFamily.bubble,
        "iso": ElementFamily.iso,
    }

    # Family names that are valid on non-interval cells
    if cell != "interval":
        families.update(
            {
                "RT": ElementFamily.RT,
                "Raviart-Thomas": ElementFamily.RT,
                "N1F": ElementFamily.RT,
                "N1div": ElementFamily.RT,
                "Nedelec 1st kind H(div)": ElementFamily.RT,
                "N1E": ElementFamily.N1E,
                "N1curl": ElementFamily.N1E,
                "Nedelec 1st kind H(curl)": ElementFamily.N1E,
                "BDM": ElementFamily.BDM,
                "Brezzi-Douglas-Marini": ElementFamily.BDM,
                "N2F": ElementFamily.BDM,
                "N2div": ElementFamily.BDM,
                "Nedelec 2nd kind H(div)": ElementFamily.BDM,
                "N2E": ElementFamily.N2E,
                "N2curl": ElementFamily.N2E,
                "Nedelec 2nd kind H(curl)": ElementFamily.N2E,
            }
        )

    # Family names that are valid for intervals
    if cell == "interval":
        families.update(
            {
                "DPC": ElementFamily.P,
            }
        )

    # Family names that are valid for tensor product cells
    if cell in ["interval", "quadrilateral", "hexahedron"]:
        families.update(
            {
                "Q": ElementFamily.P,
                "Serendipity": ElementFamily.serendipity,
                "serendipity": ElementFamily.serendipity,
                "S": ElementFamily.serendipity,
            }
        )

    # Family names that are valid for quads and hexes
    if cell in ["quadrilateral", "hexahedron"]:
        families.update(
            {
                "RTCF": ElementFamily.RT,
                "DPC": ElementFamily.DPC,
                "NCF": ElementFamily.RT,
                "RTCE": ElementFamily.N1E,
                "NCE": ElementFamily.N1E,
                "BDMCF": ElementFamily.BDM,
                "BDMCE": ElementFamily.N2E,
                "AAF": ElementFamily.BDM,
                "AAE": ElementFamily.N2E,
            }
        )

    # Family names that are valid for triangles and tetrahedra
    if cell in ["triangle", "tetrahedron"]:
        families.update(
            {
                "Regge": ElementFamily.Regge,
                "CR": ElementFamily.CR,
                "Crouzeix-Raviart": ElementFamily.CR,
            }
        )

    # Family names that are valid for triangles
    if cell in ["triangle"]:
        families.update(
            {
                "HHJ": ElementFamily.HHJ,
                "Hellan-Herrmann-Johnson": ElementFamily.HHJ,
            }
        )

    try:
        return families[family]
    except KeyError:
        raise ValueError(f"Unknown element family: {family} with cell type {cell}")


def string_to_lagrange_variant(variant: str) -> LagrangeVariant:
    """Convert a string to a Basix LagrangeVariant enum.

    Args:
        variant: Lagrange variant string.

    Returns:
        The Lagrange variant.
    """
    if variant.lower() == "gll":
        return LagrangeVariant.gll_warped
    elif variant.lower() == "chebyshev":
        return LagrangeVariant.chebyshev_isaac
    elif variant.lower() == "gl":
        return LagrangeVariant.gl_isaac

    if not hasattr(LagrangeVariant, variant.lower()):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(LagrangeVariant, variant.lower())


def string_to_dpc_variant(variant: str) -> DPCVariant:
    """Convert a string to a Basix DPCVariant enum.

    Args:
        variant: DPC variant as a string.

    Returns:
        The DPC variant.
    """
    if not hasattr(DPCVariant, variant.lower()):
        raise ValueError(f"Unknown variant: {variant}")
    return getattr(DPCVariant, variant.lower())
