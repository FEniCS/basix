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

const std::string FiniteElement__tabulate = R"(
Compute basis values and derivatives at set of points.

NOTE: The version of `FiniteElement::tabulate` with the basis data
as an out argument should be preferred for repeated call where
performance is critical

Args:
    nd: The order of derivatives, up to and including, to compute. Use 0 for the basis functions only.
    x: The points at which to compute the basis functions. The shape of x is (number of points, geometric dimension).

Returns:
    The basis functions (and derivatives). The shape is
    (derivative, point, basis fn index, value index).
    - The first index is the derivative, with higher derivatives are
    stored in triangular (2D) or tetrahedral (3D) ordering, ie for
    the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
    (0,2), (3,0)... The function basix::indexing::idx can be used to find the
    appropriate derivative.
    - The second index is the point index
    - The third index is the basis function index
    - The fourth index is the basis function component. Its has size
    one for scalar basis functions.
)";

const std::string FiniteElement__push_forward = R"(
Map function values from the reference to a physical cell. This
function can perform the mapping for multiple points, grouped by
points that share a common Jacobian.

Args:
    U: The function values on the reference. The indices are [Jacobian index, point index, components].
    J: The Jacobian of the mapping. The indices are [Jacobian index, J_i, J_j].
    detJ: The determinant of the Jacobian of the mapping. It has length `J.shape(0)`
    K: The inverse of the Jacobian of the mapping. The indices are [Jacobian index, K_i, K_j].

Returns:
    The function values on the cell. The indices are [Jacobian
    index, point index, components].
)";

const std::string FiniteElement__pull_back = R"(
Map function values from a physical cell to the reference

Args:
    u: The function values on the cell
    J: The Jacobian of the mapping
    detJ: The determinant of the Jacobian of the mapping
    K: The inverse of the Jacobian of the mapping

Returns:
    The function values on the reference. The indices are
    [Jacobian index, point index, components].
)";

const std::string FiniteElement__apply_dof_transformation = R"(
Apply DOF transformations to some data

NOTE: This function is designed to be called at runtime, so its
performance is critical.

Args:
    data: The data
    block_size: The number of data points per DOF
    cell_info: The permutation info for the cell

Returns:
    data: The data
)";

const std::string FiniteElement__apply_dof_transformation_to_transpose = R"(
Apply DOF transformations to some transposed data

NOTE: This function is designed to be called at runtime, so its
performance is critical.

Args:
    data: The data
    block_size: The number of data points per DOF
    cell_info: The permutation info for the cell

Returns:
    data: The data
)";

const std::string FiniteElement__apply_inverse_transpose_dof_transformation
    = R"(
Apply inverse transpose DOF transformations to some data

NOTE: This function is designed to be called at runtime, so its
performance is critical.

Args:
    data: The data
    block_size: The number of data points per DOF
    cell_info: The permutation info for the cell

Returns:
    data: The data
)";

const std::string FiniteElement__base_transformations = R"(
Get the base transformations.

The base transformations represent the effect of rotating or reflecting
a subentity of the cell on the numbering and orientation of the DOFs.
This returns a list of matrices with one matrix for each subentity
permutation in the following order:
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
    (ntranformations, ndofs, ndofs)
)";

const std::string FiniteElement__entity_transformations = R"(
Return the entity dof transformation matrices

Returns:
    The base transformations for this element. The shape is
    (ntranformations, ndofs, ndofs)
)";

const std::string FiniteElement__get_tensor_product_representation = R"(
Get the tensor product representation of this element, or throw an
error if no such factorisation exists.

The tensor product representation will be a vector of tuples. Each
tuple contains a vector of finite elements, and a vector of
integers. The vector of finite elements gives the elements on an
interval that appear in the tensor product representation. The
vector of integers gives the permutation between the numbering of
the tensor product DOFs and the number of the DOFs of this Basix
element.

Returns:
    The tensor product representation
)";

const std::string create_custom_element = R"(
Create a custom finite element

Args:
    cell_type: The cell type
    value_shape: The value shape of the element
    wcoeffs: Matrices for the kth value index containing the expansion coefficients defining a polynomial basis spanning the polynomial space for this element. Shape is (dim(finite element polyset), dim(Legendre polynomials))
    x: Interpolation points. Indices are (tdim, entity index, point index, dim)
    M: The interpolation matrices. Indices are (tdim, entity index, dof, vs, point_index, derivative)
    interpolation_nderivs: The number of derivatives that need to be used during interpolation
    map_type: The type of map to be used to map values from the reference to a cell
    sobolev_space: The underlying Sobolev space for the element
    discontinuous: Indicates whether or not this is the discontinuous version of the element
    highest_complete_degree: The highest degree n such that a Lagrange (or vector Lagrange) element of degree n is a subspace of this element
    highest_degree: The degree of a polynomial in this element's polyset
    poly_type: The type of polyset to use for this element

Returns:
    A custom finite element
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

const std::string compute_interpolation_operator = R"(
Compute a matrix that represents the interpolation between
two elements.

If the two elements have the same value size, this function returns
the interpolation between them.

If element_from has value size 1 and element_to has value size > 1,
then this function returns a matrix to interpolate from a blocked
element_from (ie multiple copies of element_from) into element_to.

If element_to has value size 1 and element_from has value size > 1,
then this function returns a matrix that interpolates the components
of element_from into copies of element_to.

NOTE: If the elements have different value sizes and both are
greater than 1, this function throws a runtime error

In order to interpolate functions between finite element spaces on
arbitrary cells, the functions must be pulled back to the reference
element (this pull back includes applying DOF transformations). The
matrix that this function returns can then be applied, then the
result pushed forward to the cell. If element_from and element_to
have the same map type, then only the DOF transformations need to be
applied, as the pull back and push forward cancel each other out.

Args:
    element_from: The element to interpolate from
    element_to: The element to interpolate to

Returns:
    Matrix operator that maps the 'from' degrees-of-freedom to
    the 'to' degrees-of-freedom. Shape is (ndofs(element_to),
    ndofs(element_from))
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
