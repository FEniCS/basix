"""Functions for creating finite elements."""

import typing
import numpy as _np
import numpy.typing as npt

from basix._basixcpp import create_element as _create_element
from basix._basixcpp import create_custom_element as _create_custom_element
from basix._basixcpp import ElementFamily as _EF
from basix._basixcpp import FiniteElement_float32, FiniteElement_float64  # type: ignore
from basix._basixcpp import (
    CellType,
    DPCVariant,
    ElementFamily,
    LagrangeVariant,
    # LatticeSimplexMethod,
    # LatticeType,
    MapType,
    # PolynomialType,
    PolysetType,
    # QuadratureType,
    SobolevSpace,
    # # __version__,
    # compute_interpolation_operator,
    # create_custom_element,
    # create_lattice,
    # geometry,
    # index
)


class FiniteElement:
    def __init__(self, e):
        self._e = e

    def tabulate(self, n: int, x: npt.NDArray) -> npt.NDArray[_np.float_]:
        """Compute basis values and derivatives at set of points.

        Note:
            The version of `FiniteElement::tabulate` with the basis data
            as an out argument should be preferred for repeated call where
            performance is critical

        Args:
            n: The order of derivatives, up to and including, to
              compute. Use 0 for the basis functions only.
            x: The points at which to compute the basis functions. The
                shape of x is (number of points, geometric dimension).

        Returns:
            The basis functions (and derivatives). The shape is
            (derivative, point, basis fn index, value index).
            - The first index is the derivative, with higher derivatives
            are stored in triangular (2D) or tetrahedral (3D) ordering,
            ie for the (x,y) derivatives in 2D: (0,0), (1,0), (0,1),
            (2,0), (1,1), (0,2), (3,0)... The function
            basix::indexing::idx can be used to find the appropriate
            derivative.
            - The second index is the point index
            - The fourth index is the basis function component. Its has size
            - The third index is the basis function index one for scalar
                basis functions.
        """
        return self._e.tabulate(n, x)

    def __eq__(self, other):
        return self._e == other._e

    def push_forward(self, U, J, detJ, K) -> npt.NDArray[_np.float_]:
        """Map function values from the reference to a physical cell.

        This function can perform the mapping for multiple points,
        grouped by points that share a common Jacobian.

        Args:
            U: The function values on the reference. The indices are
                [Jacobian index, point index, components].
            J: The Jacobian of the mapping. The indices are [Jacobian
                index, J_i, J_j].
            detJ: The determinant of the Jacobian of the mapping. It has
                length `J.shape(0)` K: The inverse of the Jacobian of the
            mapping. The indices are [Jacobian index, K_i, K_j].

        Returns:
            The function values on the cell. The indices are [Jacobian
            index, point index, components].
        """
        return self._e.push_forward(U, J, detJ, K)

    def pull_back(self, u: npt.NDArray, J: npt.NDArray,
                  detJ: npt.NDArray, K: npt.NDArray) -> npt.NDArray[_np.float_]:
        """Map function values from a physical cell to the reference.

        Args:
            u: The function values on the cell
            J: The Jacobian of the mapping
            detJ: The determinant of the Jacobian of the mapping
            K: The inverse of the Jacobian of the mapping

        Returns:
            The function values on the reference. The indices are
            [Jacobian index, point index, components].
        """
        return self._e.pull_back(u, J, detJ, K)

    def pre_apply_dof_transformation(self, data, block_size, cell_info) -> None:
        self._e.pre_apply_dof_transformation(data, block_size, cell_info)

    def post_apply_transpose_dof_transformation(self, data, block_size, cell_info) -> None:
        self._e.post_apply_transpose_dof_transformation(data, block_size, cell_info)

    def pre_apply_inverse_transpose_dof_transformation(self, data, block_size, cell_info) -> None:
        self._e.pre_apply_inverse_transpose_dof_transformation(data, block_size, cell_info)

    def base_transformations(self) -> npt.NDArray:
        return self._e.base_transformations()

    def entity_transformations(self) -> npt.NDArray:
        return self._e.entity_transformations()

    def get_tensor_product_representation(self):
        return self._e.get_tensor_product_representation()

    # def get_tensor_product_representation(self) -> typing.List[typing.Tuple[typing.List[FiniteElement],
    # typing.List[int]]]:
    #     return self._e.get_tensor_product_representation()

    @property
    def degree(self) -> int:
        return self._e.degree

    @property
    def embedded_superdegree(self) -> int:
        return self._e.embedded_superdegree

    @property
    def embedded_subdegree(self) -> int:
        return self._e.embedded_subdegree

    @property
    def cell_type(self) -> CellType:
        return self._e.cell_type

    @property
    def polyset_type(self) -> PolysetType:
        return self._e.polyset_type

    @property
    def dim(self) -> int:
        return self._e.dim

    @property
    def num_entity_dofs(self) -> int:
        return self._e.num_entity_dofs

    @property
    def entity_dofs(self) -> typing.List[typing.List[typing.List[int]]]:
        return self._e.entity_dofs

    @property
    def num_entity_closure_dofs(self) -> int:
        return self._e.num_entity_closure_dofs

    @property
    def entity_closure_dofs(self) -> typing.List[typing.List[typing.List[int]]]:
        return self._e.entity_closure_dofs

    @property
    def value_size(self) -> int:
        return self._e.value_size

    @property
    def value_shape(self) -> int:
        return self._e.value_shape

    @property
    def discontinuous(self) -> bool:
        return self._e.discontinuous

    @property
    def family(self) -> ElementFamily:
        return self._e.family

    @property
    def lagrange_variant(self) -> LagrangeVariant:
        return self._e.lagrange_variant

    @property
    def dpc_variant(self) -> DPCVariant:
        return self._e.dpc_variant

    @property
    def dof_transformations_are_permutations(self) -> bool:
        return self._e.dof_transformations_are_permutations

    @property
    def dof_transformations_are_identity(self) -> bool:
        return self._e.dof_transformations_are_identity

    @property
    def interpolation_is_identity(self) -> bool:
        return self._e.interpolation_is_identity

    @property
    def map_type(self) -> MapType:
        return self._e.map_type

    @property
    def sobolev_space(self) -> SobolevSpace:
        return self._e.sobolev_space

    @property
    def points(self):
        return self._e.points

    @property
    def interpolation_matrix(self):
        return self._e.interpolation_matrix

    @property
    def dual_matrix(self):
        return self._e.dual_matrix

    @property
    def coefficient_matrix(self):
        return self._e.coefficient_matrix

    @property
    def wcoeffs(self):
        return self._e.wcoeffs

    @property
    def M(self):
        return self._e.M

    @property
    def x(self):
        return self._e.x

    @property
    def has_tensor_product_factorisation(self) -> bool:
        return self._e.has_tensor_product_factorisation

    @property
    def interpolation_nderivs(self) -> int:
        return self._e.interpolation_nderivs

    @property
    def dof_ordering(self) -> typing.List[int]:
        return self._e.dof_ordering


def create_element(family_name: ElementFamily, cell_name: CellType, degree: int,
                   lvariant: typing.Optional[LagrangeVariant] = LagrangeVariant.unset,
                   dvariant: typing.Optional[DPCVariant] = DPCVariant.unset,
                   discontinuous: typing.Optional[bool] = False,
                   dof_ordering:  typing.Optional[list[int]] = [],
                   dtype: typing.Optional[npt.DTypeLike] = _np.float64) -> typing.Union[FiniteElement_float32,
                                                                                        FiniteElement_float64]:
    """Create a finite element.

    Args:
        family_name: Finite element family.
        cell_name: Cell shape.
        degree: Polynomial degree.
        lvariant: Lagrange variant type.
        dvariant: DPC variant type/
        discontinuous: If `True`, make element discontinuous
        dof_ordering:
        dtype: Element scalar type.

    Returns:
        A finite element.
    """
    return FiniteElement(_create_element(family_name, cell_name, degree, lvariant, dvariant,
                                         discontinuous, dof_ordering, _np.dtype(dtype).char))


def create_custom_element(cell_name: CellType, value_shape, wcoeffs, x, M, interpolation_nderivs: int, map_type,
                          sobolev_space, discontinuous: bool,
                          embedded_subdegree: int, embedded_superdegree: int,
                          poly_type: PolysetType) -> typing.Union[FiniteElement_float32, FiniteElement_float64]:
    """Create a custom finite element.

    Args:
        cell_name:
        value_shape:
        wcoeffs: A
        x:
        M:
        interpolation_nderivs:
        map_type:
        sobolev_space:
        discontinuous:
        embedded_subdegree:
        embedded_superdegree:
        poly_type:

    Returns:
        A finite element.
    """
    return FiniteElement(_create_custom_element(cell_name, value_shape, wcoeffs, x, M,
                                                interpolation_nderivs, map_type,
                                                sobolev_space, discontinuous, embedded_subdegree,
                                                embedded_superdegree, poly_type))


def string_to_family(family: str, cell: str) -> _EF:
    """Get a Basix ElementFamily enum representing the family type on the given cell.

    Args:
        family: The element family as a string.
        cell: The cell type as a string.

    Returns:
        The element family.

    """
    # Family names that are valid for all cells
    families = {
        "Lagrange": _EF.P,
        "P": _EF.P,
        "Bubble": _EF.bubble,
        "bubble": _EF.bubble,
        "iso": _EF.iso,
    }

    # Family names that are valid on non-interval cells
    if cell != "interval":
        families.update({
            "RT": _EF.RT,
            "Raviart-Thomas": _EF.RT,
            "N1F": _EF.RT,
            "N1div": _EF.RT,
            "Nedelec 1st kind H(div)": _EF.RT,
            "N1E": _EF.N1E,
            "N1curl": _EF.N1E,
            "Nedelec 1st kind H(curl)": _EF.N1E,
            "BDM": _EF.BDM,
            "Brezzi-Douglas-Marini": _EF.BDM,
            "N2F": _EF.BDM,
            "N2div": _EF.BDM,
            "Nedelec 2nd kind H(div)": _EF.BDM,
            "N2E": _EF.N2E,
            "N2curl": _EF.N2E,
            "Nedelec 2nd kind H(curl)": _EF.N2E,
        })

    # Family names that are valid for intervals
    if cell == "interval":
        families.update({
            "DPC": _EF.P,
        })

    # Family names that are valid for tensor product cells
    if cell in ["interval", "quadrilateral", "hexahedron"]:
        families.update({
            "Q": _EF.P,
            "Serendipity": _EF.serendipity,
            "serendipity": _EF.serendipity,
            "S": _EF.serendipity,
        })

    # Family names that are valid for quads and hexes
    if cell in ["quadrilateral", "hexahedron"]:
        families.update({
            "RTCF": _EF.RT,
            "DPC": _EF.DPC,
            "NCF": _EF.RT,
            "RTCE": _EF.N1E,
            "NCE": _EF.N1E,
            "BDMCF": _EF.BDM,
            "BDMCE": _EF.N2E,
            "AAF": _EF.BDM,
            "AAE": _EF.N2E,
        })

    # Family names that are valid for triangles and tetrahedra
    if cell in ["triangle", "tetrahedron"]:
        families.update({
            "Regge": _EF.Regge,
            "CR": _EF.CR,
            "Crouzeix-Raviart": _EF.CR,
        })

    # Family names that are valid for triangles
    if cell in "triangle":
        families.update({
            "HHJ": _EF.HHJ,
            "Hellan-Herrmann-Johnson": _EF.HHJ,
        })

    if family in families:
        return families[family]

    raise ValueError(f"Unknown element family: {family} with cell type {cell}")
