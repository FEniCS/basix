#include <string>

namespace basix::docstring
{

const std::string topology = R"(
Cell topology

Args:
    celltype: Cell Type

Returns::
    List of topology (vertex indices) for each dimension (0..tdim)
)";

const std::string geometry = R"(
Cell geometry

Args:
    celltype: Cell Type

Returns::
    (0) Vertex point data of the cell and (1) the shape of the
    data array. The points are stored in row-major format and the shape
    is is (npoints, gdim)
)";

const std::string sub_entity_connectivity = R"(
Get the numbers of entities connected to each subentity of the cell.

Returns a vector of the form: output[dim][entity_n][connected_dim] =
[connected_entity_n0, connected_entity_n1, ...] This indicates that
the entity of dimension `dim` and number `entity_n` is connected to
the entities of dimension `connected_dim` and numbers
`connected_entity_n0`, `connected_entity_n1`, ...

Args:
    celltype: Cell Type

Returns:
    List of topology (vertex indices) for each dimension (0..tdim)
)";

const std::string sub_entity_geometry = R"(
Sub-entity of a cell, given by topological dimension and index

Args:
    celltype: The cell::type
    dim: Dimension of sub-entity
    index: Local index of sub-entity

Returns:
    Set of vertex points of the sub-entity. Shape is (npoints, gdim)
)";

const std::string create_lattice__celltype_n_type_exterior = R"(
Create a lattice of points on a reference cell optionally
including the outer surface points.

For a given `celltype`, this creates a set of points on a regular
grid which covers the cell, eg for a quadrilateral, with n=2, the
points are: `[0,0], [0.5,0], [1,0], [0,0.5], [0.5,0.5], [1,0.5],
[0,1], [0.5,1], [1,1]`. If the parameter exterior is set to false,
the points lying on the external boundary are omitted, in this case
for a quadrilateral with `n == 2`, the points are: `[0.5, 0.5]`. The
lattice type can be chosen as type::equispaced or type::gll. The
type::gll lattice has points spaced along each edge at the
Gauss-Lobatto-Legendre quadrature points. These are the same as
type::equispaced when `n < 3`.

Args:
    celltype: The cell type
    n: Size in each direction. There are `n + 1` points along each edge of the cell
    type: A lattice type
    exterior: If set, includes outer boundaries

Returns:
    Set of points. Shape is `(npoints, tdim)` and storage is
    row-major
)";

const std::string create_lattice__celltype_n_type_exterior_method = R"(
Create a lattice of points on a reference cell optionally
including the outer surface points.

For a given `celltype`, this creates a set of points on a regular
grid which covers the cell, eg for a quadrilateral, with n=2, the
points are: `[0,0], [0.5,0], [1,0], [0,0.5], [0.5,0.5], [1,0.5],
[0,1], [0.5,1], [1,1]`. If the parameter exterior is set to false,
the points lying on the external boundary are omitted, in this case
for a quadrilateral with `n == 2`, the points are: `[0.5, 0.5]`. The
lattice type can be chosen as type::equispaced or type::gll. The
type::gll lattice has points spaced along each edge at the
Gauss-Lobatto-Legendre quadrature points. These are the same as
type::equispaced when `n < 3`.

Args:
    celltype: The cell type
    n: Size in each direction. There are `n + 1` points along each edge of the cell
    type: A lattice type
    exterior: If set, includes outer boundaries
    simplex_method: The method used to generate points on simplices

Returns:
    Set of points. Shape is `(npoints, tdim)` and storage is
    row-major
)";

const std::string cell_volume = R"(
Get the volume of a reference cell

Args:
    cell_type: Type of cell

Returns:
    The volume of the cell
)";

const std::string cell_facet_normals = R"(
Get the normals to the facets of a reference cell oriented using the
low-to-high ordering of the facet

Args:
    cell_type: Type of cell

Returns:
    The normals. Shape is (nfacets, gdim)
)";

const std::string cell_facet_reference_volumes = R"(
Get the reference volumes of the facets of a reference cell

Args:
    cell_type: Type of cell

Returns:
    The volumes of the references associated with each facet
)";

const std::string cell_facet_outward_normals = R"(
Get the (outward) normals to the facets of a reference cell

Args:
    cell_type: Type of cell

Returns:
    The outward normals. Shape is (nfacets, gdim)
)";

const std::string cell_facet_orientations = R"(
Get an array of bools indicating whether or not the facet normals are
outward pointing

Args:
    cell_type: Type of cell

Returns:
    The orientations
)";

const std::string cell_facet_jacobians = R"(
Get the jacobians of the facets of a reference cell

Args:
    cell_type: Type of cell

Returns:
    The jacobians of the facets. Shape is (nfacets, gdim, gdim - 1)
)";

const std::string
    create_element__family_cell_degree_lvariant_dvariant_discontinuous_dof_ordering
    = R"(
Create an element using a given Lagrange variant and a given DPC variant

Args:
    family: The element family
    cell: The reference cell type that the element is defined on
    degree: The degree of the element
    lvariant: The variant of Lagrange to use
    dvariant: The variant of DPC to use
    discontinuous: Indicates whether the element is discontinuous between cells points of the element. The discontinuous element will have the same DOFs, but they will all be associated with the interior of the cell.
    dof_ordering: Ordering of dofs for ElementDofLayout

Returns:
    A finite element
)";

const std::string tabulate_polynomial_set = R"(
Tabulate the orthonormal polynomial basis, and derivatives,
at points on the reference cell.

All derivatives up to the given order are computed. If derivatives
are not required, use `n = 0`. For example, order `n = 2` for a 2D
cell, will compute the basis \f$\psi, d\psi/dx, d\psi/dy, d^2
\psi/dx^2, d^2\psi/dxdy, d^2\psi/dy^2\f$ in that order (0, 0), (1,
0), (0, 1), (2, 0), (1, 1), (0 ,2).

For an interval cell there are `nderiv + 1` derivatives, for a 2D
cell, there are `(nderiv + 1)(nderiv + 2)/2` derivatives, and in 3D,
there are `(nderiv + 1)(nderiv + 2)(nderiv + 3)/6`. The ordering is
'triangular' with the lower derivatives appearing first.

Args:
    celltype: Cell type
    ptype: The polynomial type
    d: Polynomial degree
    n: Maximum derivative order. Use n = 0 for the basis only.
    x: Points at which to evaluate the basis. The shape is (number of points, geometric dimension).

Returns:
    Polynomial sets, for each derivative, tabulated at points.
    The shape is `(number of derivatives computed, number of points,
    basis index)`.

    - The first index is the derivative. The first entry is the basis
    itself. Derivatives are stored in triangular (2D) or tetrahedral
    (3D) ordering, eg if `(p, q)` denotes `p` order derivative with
    respect to `x` and `q` order derivative with respect to `y`, [0] ->
    (0, 0), [1] -> (1, 0), [2] -> (0, 1), [3] -> (2, 0), [4] -> (1, 1),
    [5] -> (0, 2), [6] -> (3, 0),...
    The function basix::indexing::idx maps tuples `(p, q, r)` to the
    array index.

    - The second index is the point, with index `i` corresponding to the
    point in row `i` of @p x.

    - The third index is the basis function index.
    TODO: Does the order for the third index need to be documented?
)";

const std::string tabulate_polynomials = R"(
Tabulate a set of polynomials.

Args:
    polytype: Polynomial type
    celltype: Cell type
    d: Polynomial degree
    x: Points at which to evaluate the basis. The shape is (number of points, geometric dimension).

Returns:
    Polynomial sets, for each derivative, tabulated at points.
    The shape is `(basis index, number of points)`.
)";

const std::string polynomials_dim = R"(
Dimension of a polynomial space.

Args:
    polytype: The polynomial type
    cell: The cell type
    d: The polynomial degree

Returns:
    The number terms in the basis spanning a space of
    polynomial degree @p d
)";

const std::string make_quadrature__rule_celltype_polytype_m = R"(
Make a quadrature rule on a reference cell

Args:
    rule: Type of quadrature rule (or use quadrature::Default)
    celltype: The cell type
    polytype: The polyset type
    m: Maximum degree of polynomial that this quadrature rule will integrate exactly

Returns:
    List of points and list of weights. The number of points
    arrays has shape (num points, gdim)
)";

const std::string index__p = R"(
Compute trivial indexing in a 1D array (for completeness)

Args:
    p: Index in x

Returns:
    1D Index
)";

const std::string index__p_q = R"(
Compute indexing in a 2D triangular array compressed into a 1D
array. This can be used to find the index of a derivative returned
by `FiniteElement::tabulate`. For instance to find d2N/dx2, use
`FiniteElement::tabulate(2, points)[idx(2, 0)];`

Args:
    p: Index in x
    q: Index in y

Returns:
    1D Index
)";

const std::string index__p_q_r = R"(
Compute indexing in a 3D tetrahedral array compressed into a 1D
array

Args:
    p: Index in x
    q: Index in y
    r: Index in z

Returns:
    1D Index
)";

const std::string space_intersection = R"(
Get the intersection of two Sobolev spaces.

Args:
    space1: First space
    space2: Second space

Returns:
    Intersection of the spaces
)";

const std::string superset = R"(
Get the polyset types that is a superset of two types on the given
cell

Args:
    cell: The cell type
    type1: The first polyset type
    type2: The second polyset type

Returns::
    The superset type
)";

const std::string restriction = R"(
Get the polyset type that represents the restrictions of a type on a
subentity

Args:
    ptype: The polyset type
    cell: The cell type
    restriction_cell: The cell type of the subentity

Returns::
    The restricted polyset type
)";

} // namespace basix::docstring
