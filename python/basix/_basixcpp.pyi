from collections.abc import Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray, ArrayLike


def topology(arg: CellType, /) -> list[list[list[int]]]: ...

def geometry(arg: CellType, /) -> NDArray[numpy.float64]: ...

def sub_entity_type(arg0: CellType, arg1: int, arg2: int, /) -> CellType: ...

def sub_entity_connectivity(arg: CellType, /) -> list[list[list[list[int]]]]: ...

def sub_entity_geometry(arg0: CellType, arg1: int, arg2: int, /) -> NDArray[numpy.float64]: ...

def subentity_types(arg: CellType, /) -> list[list[CellType]]: ...

def sobolev_space_intersection(arg0: SobolevSpace, arg1: SobolevSpace, /) -> SobolevSpace: ...

class LatticeType(enum.IntEnum):
    """Lattice type."""

    equispaced = 0

    gll = 1

    chebyshev = 2

    gl = 4

class LatticeSimplexMethod(enum.IntEnum):
    """Lattice simplex method."""

    none = 0

    warp = 1

    isaac = 2

    centroid = 3

class PolynomialType(enum.IntEnum):
    """Polynomial type."""

    legendre = 0

    lagrange = 1

    bernstein = 2

def tabulate_polynomials(arg0: PolynomialType, arg1: CellType, arg2: int, arg3: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], /) -> NDArray[numpy.float64]: ...

def polynomials_dim(arg0: PolynomialType, arg1: CellType, arg2: int, /) -> int: ...

def create_lattice(arg0: CellType, arg1: int, arg2: LatticeType, arg3: bool, arg4: LatticeSimplexMethod, /) -> NDArray[numpy.float64]: ...

class MapType(enum.IntEnum):
    """Element map type."""

    identity = 0

    L2Piola = 1

    covariantPiola = 2

    contravariantPiola = 3

    doubleCovariantPiola = 4

    doubleContravariantPiola = 5

class SobolevSpace(enum.IntEnum):
    """Sobolev space."""

    L2 = 0

    H1 = 1

    H2 = 2

    H3 = 3

    HInf = 8

    HDiv = 10

    HCurl = 11

    HEin = 12

    HDivDiv = 13

class QuadratureType(enum.IntEnum):
    """Quadrature type."""

    default = 0

    gauss_jacobi = 1

    gll = 2

    xiao_gimbutas = 3

class CellType(enum.IntEnum):
    """Cell type."""

    point = 0

    interval = 1

    triangle = 2

    tetrahedron = 3

    quadrilateral = 4

    hexahedron = 5

    prism = 6

    pyramid = 7

def cell_volume(arg: CellType, /) -> float: ...

def cell_facet_normals(arg: CellType, /) -> NDArray[numpy.float64]: ...

def cell_facet_reference_volumes(arg: CellType, /) -> NDArray[numpy.float64]: ...

def cell_facet_outward_normals(arg: CellType, /) -> NDArray[numpy.float64]: ...

def cell_facet_orientations(arg: CellType, /) -> list[int]: ...

def cell_facet_jacobians(arg: CellType, /) -> NDArray[numpy.float64]: ...

def cell_edge_jacobians(arg: CellType, /) -> NDArray[numpy.float64]: ...

class ElementFamily(enum.IntEnum):
    """Finite element family."""

    custom = 0

    P = 1

    BDM = 4

    RT = 2

    N1E = 3

    N2E = 5

    Regge = 7

    HHJ = 11

    bubble = 9

    serendipity = 10

    DPC = 8

    CR = 6

    Hermite = 12

    iso = 13

class LagrangeVariant(enum.IntEnum):
    """Lagrange element variant."""

    unset = 0

    equispaced = 1

    gll_warped = 2

    gll_isaac = 3

    gll_centroid = 4

    chebyshev_warped = 5

    chebyshev_isaac = 6

    chebyshev_centroid = 7

    gl_warped = 8

    gl_isaac = 9

    gl_centroid = 10

    legendre = 11

    bernstein = 12

class DPCVariant(enum.IntEnum):
    """DPC variant."""

    unset = 0

    simplex_equispaced = 1

    simplex_gll = 2

    horizontal_equispaced = 3

    horizontal_gll = 4

    diagonal_equispaced = 5

    diagonal_gll = 6

    legendre = 7

def create_element(arg0: ElementFamily, arg1: CellType, arg2: int, arg3: LagrangeVariant, arg4: DPCVariant, arg5: bool, arg6: Sequence[int], arg7: str, /) -> FiniteElement_float32 | FiniteElement_float64: ...

def create_tp_element(arg0: ElementFamily, arg1: CellType, arg2: int, arg3: LagrangeVariant, arg4: DPCVariant, arg5: bool, arg6: str, /) -> FiniteElement_float32 | FiniteElement_float64: ...

def tp_factors(arg0: ElementFamily, arg1: CellType, arg2: int, arg3: LagrangeVariant, arg4: DPCVariant, arg5: bool, arg6: Sequence[int], arg7: str, /) -> list[list[FiniteElement_float32]] | list[list[FiniteElement_float64]]: ...

def tp_dof_ordering(arg0: ElementFamily, arg1: CellType, arg2: int, arg3: LagrangeVariant, arg4: DPCVariant, arg5: bool, /) -> list[int]: ...

def lex_dof_ordering(arg0: ElementFamily, arg1: CellType, arg2: int, arg3: LagrangeVariant, arg4: DPCVariant, arg5: bool, /) -> list[int]: ...

class PolysetType(enum.IntEnum):
    """Polyset type."""

    standard = 0

    macroedge = 1

def superset(arg0: CellType, arg1: PolysetType, arg2: PolysetType, /) -> PolysetType: ...

def restriction(arg0: PolysetType, arg1: CellType, arg2: CellType, /) -> PolysetType: ...

def make_quadrature(arg0: QuadratureType, arg1: CellType, arg2: PolysetType, arg3: int, /) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]: ...

def gauss_jacobi_rule(arg0: float, arg1: int, /) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]: ...

@overload
def index(arg: int, /) -> int: ...

@overload
def index(arg0: int, arg1: int, /) -> int: ...

@overload
def index(arg0: int, arg1: int, arg2: int, /) -> int: ...

class FiniteElement_float32:
    def tabulate(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', writable=False)], /) -> NDArray[numpy.float32]: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def hash(self) -> int: ...

    @overload
    def permute_subentity_closure(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, /) -> None: ...

    @overload
    def permute_subentity_closure(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, arg3: int, /) -> None: ...

    @overload
    def permute_subentity_closure_inv(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, /) -> None: ...

    @overload
    def permute_subentity_closure_inv(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, arg3: int, /) -> None: ...

    def push_forward(self, arg0: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', writable=False)], arg1: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', writable=False)], arg2: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', writable=False)], arg3: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', writable=False)], /) -> NDArray[numpy.float32]: ...

    def pull_back(self, arg0: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', writable=False)], arg1: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', writable=False)], arg2: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', writable=False)], arg3: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', writable=False)], /) -> NDArray[numpy.float32]: ...

    def T_apply(self, arg0: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C')], arg1: int, arg2: int, /) -> None: ...

    def Tt_apply_right(self, arg0: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C')], arg1: int, arg2: int, /) -> None: ...

    def Tt_inv_apply(self, arg0: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C')], arg1: int, arg2: int, /) -> None: ...

    def base_transformations(self) -> NDArray[numpy.float32]: ...

    def entity_transformations(self) -> dict: ...

    def get_tensor_product_representation(self) -> list[list[FiniteElement_float32]]: ...

    @property
    def degree(self) -> int: ...

    @property
    def embedded_superdegree(self) -> int: ...

    @property
    def embedded_subdegree(self) -> int: ...

    @property
    def cell_type(self) -> CellType: ...

    @property
    def polyset_type(self) -> PolysetType: ...

    @property
    def dim(self) -> int: ...

    @property
    def num_entity_dofs(self) -> list[list[int]]: ...

    @property
    def entity_dofs(self) -> list[list[list[int]]]: ...

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]: ...

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]: ...

    @property
    def value_size(self) -> int: ...

    @property
    def value_shape(self) -> list[int]: ...

    @property
    def discontinuous(self) -> bool: ...

    @property
    def family(self) -> ElementFamily: ...

    @property
    def lagrange_variant(self) -> LagrangeVariant: ...

    @property
    def dpc_variant(self) -> DPCVariant: ...

    @property
    def dof_transformations_are_permutations(self) -> bool: ...

    @property
    def dof_transformations_are_identity(self) -> bool: ...

    @property
    def interpolation_is_identity(self) -> bool: ...

    @property
    def map_type(self) -> MapType: ...

    @property
    def sobolev_space(self) -> SobolevSpace: ...

    @property
    def points(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), writable=False)]: ...

    @property
    def interpolation_matrix(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), writable=False)]: ...

    @property
    def dual_matrix(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), writable=False)]: ...

    @property
    def coefficient_matrix(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), writable=False)]:
        """Coefficient matrix."""

    @property
    def wcoeffs(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None), writable=False)]: ...

    @property
    def M(self) -> list[list[Annotated[NDArray[numpy.float32], dict(writable=False)]]]: ...

    @property
    def x(self) -> list[list[Annotated[NDArray[numpy.float32], dict(writable=False)]]]: ...

    @property
    def has_tensor_product_factorisation(self) -> bool: ...

    @property
    def interpolation_nderivs(self) -> int: ...

    @property
    def dof_ordering(self) -> list[int]: ...

    @property
    def dtype(self) -> str: ...

def create_custom_element_float32(cell_type: CellType, value_shape: Sequence[int], wcoeffs: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', writable=False)], x: Sequence[Sequence[Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', writable=False)]]], M: Sequence[Sequence[Annotated[NDArray[numpy.float32], dict(shape=(None, None, None, None), order='C', writable=False)]]], interpolation_nderivs: int, map_type: MapType, sobolev_space: SobolevSpace, discontinuous: bool, embedded_subdegree: int, embedded_superdegree: int, poly_type: PolysetType) -> FiniteElement_float32: ...

@overload
def compute_interpolation_operator(arg0: FiniteElement_float32, arg1: FiniteElement_float32, /) -> NDArray[numpy.float32]: ...

@overload
def compute_interpolation_operator(arg0: FiniteElement_float64, arg1: FiniteElement_float64, /) -> NDArray[numpy.float64]: ...

def tabulate_polynomial_set(celltype: CellType, polytype: PolysetType, d: int, n: int, x: Annotated[ArrayLike, dict(dtype='float64', writable=False, shape=(None, None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', )]: ...

class FiniteElement_float64:
    def tabulate(self, arg0: int, arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], /) -> NDArray[numpy.float64]: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def hash(self) -> int: ...

    @overload
    def permute_subentity_closure(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, /) -> None: ...

    @overload
    def permute_subentity_closure(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, arg3: int, /) -> None: ...

    @overload
    def permute_subentity_closure_inv(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, /) -> None: ...

    @overload
    def permute_subentity_closure_inv(self, arg0: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C')], arg1: int, arg2: CellType, arg3: int, /) -> None: ...

    def push_forward(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), order='C', writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), order='C', writable=False)], arg2: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], arg3: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), order='C', writable=False)], /) -> NDArray[numpy.float64]: ...

    def pull_back(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), order='C', writable=False)], arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), order='C', writable=False)], arg2: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], arg3: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), order='C', writable=False)], /) -> NDArray[numpy.float64]: ...

    def T_apply(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], arg1: int, arg2: int, /) -> None: ...

    def Tt_apply_right(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], arg1: int, arg2: int, /) -> None: ...

    def Tt_inv_apply(self, arg0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], arg1: int, arg2: int, /) -> None: ...

    def base_transformations(self) -> NDArray[numpy.float64]: ...

    def entity_transformations(self) -> dict: ...

    def get_tensor_product_representation(self) -> list[list[FiniteElement_float64]]: ...

    @property
    def degree(self) -> int: ...

    @property
    def embedded_superdegree(self) -> int: ...

    @property
    def embedded_subdegree(self) -> int: ...

    @property
    def cell_type(self) -> CellType: ...

    @property
    def polyset_type(self) -> PolysetType: ...

    @property
    def dim(self) -> int: ...

    @property
    def num_entity_dofs(self) -> list[list[int]]: ...

    @property
    def entity_dofs(self) -> list[list[list[int]]]: ...

    @property
    def num_entity_closure_dofs(self) -> list[list[int]]: ...

    @property
    def entity_closure_dofs(self) -> list[list[list[int]]]: ...

    @property
    def value_size(self) -> int: ...

    @property
    def value_shape(self) -> list[int]: ...

    @property
    def discontinuous(self) -> bool: ...

    @property
    def family(self) -> ElementFamily: ...

    @property
    def lagrange_variant(self) -> LagrangeVariant: ...

    @property
    def dpc_variant(self) -> DPCVariant: ...

    @property
    def dof_transformations_are_permutations(self) -> bool: ...

    @property
    def dof_transformations_are_identity(self) -> bool: ...

    @property
    def interpolation_is_identity(self) -> bool: ...

    @property
    def map_type(self) -> MapType: ...

    @property
    def sobolev_space(self) -> SobolevSpace: ...

    @property
    def points(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)]: ...

    @property
    def interpolation_matrix(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)]: ...

    @property
    def dual_matrix(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)]: ...

    @property
    def coefficient_matrix(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)]:
        """Coefficient matrix."""

    @property
    def wcoeffs(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)]: ...

    @property
    def M(self) -> list[list[Annotated[NDArray[numpy.float64], dict(writable=False)]]]: ...

    @property
    def x(self) -> list[list[Annotated[NDArray[numpy.float64], dict(writable=False)]]]: ...

    @property
    def has_tensor_product_factorisation(self) -> bool: ...

    @property
    def interpolation_nderivs(self) -> int: ...

    @property
    def dof_ordering(self) -> list[int]: ...

    @property
    def dtype(self) -> str: ...

def create_custom_element_float64(cell_type: CellType, value_shape: Sequence[int], wcoeffs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], x: Sequence[Sequence[Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)]]], M: Sequence[Sequence[Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None), order='C', writable=False)]]], interpolation_nderivs: int, map_type: MapType, sobolev_space: SobolevSpace, discontinuous: bool, embedded_subdegree: int, embedded_superdegree: int, poly_type: PolysetType) -> FiniteElement_float64: ...
