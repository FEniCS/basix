#include <string>

namespace basix::docstring
{

const std::string e_lagrange__create_lagrange
    = R"(Create a Lagrange element on cell with given degree

Parameters
=========
celltype
    The reference cell type that the element is defined on
degree
    The degree of the element
variant
    The variant of the element to be created
discontinuous
points
    of the element

Returns
=======
A FiniteElement
)";

const std::string e_lagrange__create_dpc
    = R"(Create a DPC element on cell with given degree

Parameters
=========
celltype
    The reference cell type that the element is defined on
degree
    The degree of the element
discontinuous
    TODO: document this

Returns
=======
A FiniteElement
)";

const std::string dof_transformations__interval_reflection
    = R"(Reflect the DOFs on an interval

Parameters
=========
degree
    The number of DOFs on the interval

Returns
=======
A reordering of the numbers 0 to degree-1 representing the
transformation
)";

const std::string dof_transformations__triangle_reflection
    = R"(Reflect the DOFs on a triangle

Parameters
=========
degree
    The number of DOFs along one side of the triangle

Returns
=======
A reordering of the numbers 0 to (degree)*(degree+1)/2-1
representing the transformation
)";

const std::string dof_transformations__triangle_rotation
    = R"(Rotate the DOFs on a triangle

Parameters
=========
degree
    The number of DOFs along one side of the triangle

Returns
=======
A reordering of the numbers 0 to (degree)*(degree+1)/2-1
representing the transformation
)";

const std::string dof_transformations__quadrilateral_reflection
    = R"(Reflect the DOFs on a quadrilateral

Parameters
=========
degree
    The number of DOFs along one side of the quadrilateral

Returns
=======
A reordering of the numbers 0 to degree*degree-1 representing the
transformation
)";

const std::string dof_transformations__quadrilateral_rotation
    = R"(Rotate the DOFs on a quadrilateral

Parameters
=========
degree
    The number of DOFs along one side of the quadrilateral

Returns
=======
A reordering of the numbers 0 to degree*degree-1 representing the
transformation
)";

const std::string e_brezzi_douglas_marini__create_bdm = R"(Create BDM element

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string e_crouzeix_raviart__create_cr = R"(Crouzeix-Raviart element
NOTE: degree must be 1 for Crouzeix-Raviart

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string finite_element__basix = R"(Placeholder
)";

const std::string finite_element__compute_expansion_coefficients
    = R"(Calculates the basis functions of the finite element, in terms of the
polynomial basis.

The below explanation uses Einstein notation.

The basis functions ${\phi_i}$ of a finite element are represented
as a linear combination of polynomials $\{p_j\}$ in an underlying
polynomial basis that span the space of all d-dimensional polynomials up
to order $k \ (P_k^d)$:
\f[ \phi_i = c_{ij} p_j \f]

In some cases, the basis functions $\{\phi_i\}$ do not span the
full space $P_k$, in which case we denote space spanned by the
basis functions by $\{q_k\}$, which can be represented by:
\[  q_i = b_{ij} p_j. \]
This leads to
\[  \phi_i = c^{\prime}_{ij} q_j = c^{\prime}_{ij} b_{jk} p_k,  \]
and in matrix form:
\f[
\phi = C^{\prime} B p
\f]

If the basis functions span the full space, then $ B $ is simply
the identity.

The basis functions $\phi_i$ are defined by a dual set of functionals
$\{f_i\}$. The basis functions are the functions in span{$q_k$} such
that
\[ f_i(\phi_j) = \delta_{ij} \]
and inserting the expression for $\phi_{j}$:
\[ f_i(c^{\prime}_{jk}b_{kl}p_{l}) = c^{\prime}_{jk} b_{kl} f_i \left(
p_{l} \right) \]

Defining a matrix D given by applying the functionals to each
polynomial $p_j$:
\[ [D] = d_{ij},\mbox{ where } d_{ij} = f_i(p_j), \]
we have:
\[ C^{\prime} B D^{T} = I \]

and

\[ C^{\prime} = (B D^{T})^{-1}. \]

Recalling that $C = C^{\prime} B$, where $C$ is the matrix
form of $c_{ij}$,

\[ C = (B D^{T})^{-1} B \]

This function takes the matrices B (span_coeffs) and D (dual) as
inputs and returns the matrix C.

Example: Order 1 Lagrange elements on a triangle
------------------------------------------------
On a triangle, the scalar expansion basis is:
\[ p_0 = \sqrt{2}/2 \qquad
p_1 = \sqrt{3}(2x + y - 1) \qquad
p_2 = 3y - 1 \]
These span the space $P_1$.

Lagrange order 1 elements span the space P_1, so in this example,
B (span_coeffs) is the identity matrix:
\[ B = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \end{bmatrix} \]

The functionals defining the Lagrange order 1 space are point
evaluations at the three vertices of the triangle. The matrix D
(dual) given by applying these to p_0 to p_2 is:
\[ \mbox{dual} = \begin{bmatrix}
\sqrt{2}/2 &  -\sqrt{3} & -1 \\
\sqrt{2}/2 &   \sqrt{3} & -1 \\
\sqrt{2}/2 &          0 &  2 \end{bmatrix} \]

For this example, this function outputs the matrix:
\[ C = \begin{bmatrix}
\sqrt{2}/3 & -\sqrt{3}/6 &  -1/6 \\
\sqrt{2}/3 & \sqrt{3}/6  &  -1/6 \\
\sqrt{2}/3 &          0  &   1/3 \end{bmatrix} \]
The basis functions of the finite element can be obtained by applying
the matrix C to the vector $[p_0, p_1, p_2]$, giving:
\[ \begin{bmatrix} 1 - x - y \\ x \\ y \end{bmatrix} \]

Example: Order 1 Raviart-Thomas on a triangle
---------------------------------------------
On a triangle, the 2D vector expansion basis is:
\[ \begin{matrix}
p_0 & = & (\sqrt{2}/2, 0) \\
p_1 & = & (\sqrt{3}(2x + y - 1), 0) \\
p_2 & = & (3y - 1, 0) \\
p_3 & = & (0, \sqrt{2}/2) \\
p_4 & = & (0, \sqrt{3}(2x + y - 1)) \\
p_5 & = & (0, 3y - 1)
\end{matrix}
\]
These span the space $ P_1^2 $.

Raviart-Thomas order 1 elements span a space smaller than $ P_1^2 $,
so B (span_coeffs) is not the identity. It is given by:
\[ B = \begin{bmatrix}
1 &  0 &  0 &    0 &  0 &   0 \\
0 &  0 &  0 &    1 &  0 &     0 \\
1/12 &  \sqrt{6}/48 &  -\sqrt{2}/48 &  1/12 &  0 &  \sqrt{2}/24
\end{bmatrix}
\]
Applying the matrix B to the vector $[p_0, p_1, ..., p_5]$ gives the
basis of the polynomial space for Raviart-Thomas:
\[ \begin{bmatrix}
\sqrt{2}/2 &  0 \\
0 &  \sqrt{2}/2 \\
\sqrt{2}x/8  & \sqrt{2}y/8
\end{bmatrix} \]

The functionals defining the Raviart-Thomas order 1 space are integral
of the normal components along each edge. The matrix D (dual) given
by applying these to $p_0$ to $p_5$ is:
\[ D = \begin{bmatrix}
-\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 & -\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 \\
-\sqrt{2}/2 &  \sqrt{3}/2 & -1/2 &          0  &          0 &    0 \\
0 &         0   &    0 &  \sqrt{2}/2 &          0 &   -1
\end{bmatrix} \]

In this example, this function outputs the matrix:
\[  C = \begin{bmatrix}
-\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 & -\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 \\
-\sqrt{2}/2 &  \sqrt{3}/2 & -1/2 &          0  &          0  &    0 \\
0 &          0  &    0 &  \sqrt{2}/2 &          0  &   -1
\end{bmatrix} \]
The basis functions of the finite element can be obtained by applying
the matrix C to the vector $[p_0, p_1, ..., p_5]$, giving:
\[ \begin{bmatrix}
-x & -y \\
x - 1 & y \\
-x & 1 - y \end{bmatrix} \]

Parameters
=========
cell_type
    The cells shape
B
    Matrices for the kth value index containing the
    expansion coefficients defining a polynomial basis spanning the
    polynomial space for this element
M
    The interpolation tensor, such that the dual matrix
    \f$D\f$ is computed by \f$D = MP\f$
x
    The interpolation points. The vector index is for
    points on entities of the same dimension, ordered with the lowest
    topological dimension being first. Each 3D tensor hold the points on
    cell entities of a common dimension. The shape of the 3d tensors is
    (num_entities, num_points_per_entity, tdim).
degree
    The degree of the polynomial basis P used to
    create the element (before applying B)

Returns
=======
The matrix C of expansion coefficients that define the basis
functions of the finite element space. The shape is (num_dofs,
value_size, basis_dim)
)";

const std::string finite_element__make_discontinuous
    = R"(Creates a version of the interpolation points, interpolation matrices and
entity transformation that represent a discontinuous version of the element.
This discontinuous version will have the same DOFs but they will all be
associated with the interior of the reference cell.

Parameters
=========
x
    Interpolation points. Shape is (tdim, entity index,
    point index, dim)
M
    The interpolation matrices. Indices are (tdim, entity
    index, dof, vs, point_index)
entity_transformations
    Entity transformations
tdim
    The topological dimension of the cell the element is defined
    on
value_size
    The value size of the element
)";

const std::string finite_element__cell_type = R"(Get the element cell type

Returns
=======
The cell type
)";

const std::string finite_element__degree = R"(Get the element polynomial degree

Returns
=======
Polynomial degree
)";

const std::string finite_element__value_size = R"(Get the element value size
This is just a convenience function returning product(value_shape)

Returns
=======
Value size
)";

const std::string finite_element__value_shape
    = R"(Get the element value tensor shape, e.g. returning [1] for scalars.

Returns
=======
Value shape
)";

const std::string finite_element__dim
    = R"(Dimension of the finite element space (number of degrees of
freedom for the element)

Returns
=======
Number of degrees of freedom
)";

const std::string finite_element__family = R"(Get the finite element family

Returns
=======
The family
)";

const std::string finite_element__mapping_type
    = R"(Get the mapping type used for this element

Returns
=======
The mapping
)";

const std::string finite_element__discontinuous
    = R"(Indicates whether this element is the discontinuous variant

Returns
=======
True if this element is a discontinuous version
of the element
)";

const std::string finite_element__dof_transformations_are_permutations
    = R"(Indicates whether the dof transformations are all permutations

Returns
=======
True or False
)";

const std::string finite_element__dof_transformations_are_identity
    = R"(Indicates whether the dof transformations are all the identity

Returns
=======
True or False
)";

const std::string finite_element__map_push_forward
    = R"(Map function values from the reference to a physical cell. This
function can perform the mapping for multiple points, grouped by
points that share a common Jacobian.

Parameters
=========
U
    The function values on the reference. The indices are
    [Jacobian index, point index, components].
J
    The Jacobian of the mapping. The indices are [Jacobian
    index, J_i, J_j].
detJ
    The determinant of the Jacobian of the mapping. It has
    length `J.shape(0)`
K
    The inverse of the Jacobian of the mapping. The indices
    are [Jacobian index, K_i, K_j].

Returns
=======
The function values on the cell. The indices are [Jacobian
index, point index, components].
)";

const std::string finite_element__map_push_forward_m
    = R"(Direct to memory push forward

Parameters
=========
U
    Data defined on the reference element. It must have
    dimension 3. The first index is for the geometric/map data, the
    second is the point index for points that share map data, and the
    third index is (vector) component, e.g. `u[i,:,:]` are points that
    are mapped by `J[i,:,:]`.
J
    The Jacobians. It must have dimension 3. The first
    index is for the ith Jacobian, i.e. J[i,:,:] is the ith Jacobian.
detJ
    The determinant of J. `detJ[i]` is equal to
    `det(J[i,:,:])`. It must have dimension 1.
K
    The
    inverse of J, `K[i,:,:] = J[i,:,:]^-1`. It must
    have dimension 3.
u
    The input `U` mapped to the physical. It must have
    dimension 3.
)";

const std::string finite_element__map_pull_back
    = R"(Map function values from a physical cell to the reference

Parameters
=========
u
    The function values on the cell
J
    The Jacobian of the mapping
detJ
    The determinant of the Jacobian of the mapping
K
    The inverse of the Jacobian of the mapping

Returns
=======
The function values on the reference
)";

const std::string finite_element__map_pull_back_m
    = R"(Map function values from a physical cell back to to the reference

Parameters
=========
u
    Data defined on the physical element. It must have
    dimension 3. The first index is for the geometric/map data, the
    second is the point index for points that share map data, and the
    third index is (vector) component, e.g. `u[i,:,:]` are points that
    are mapped by `J[i,:,:]`.
J
    The Jacobians. It must have dimension 3. The first
    index is for the ith Jacobian, i.e. J[i,:,:] is the ith Jacobian.
detJ
    The determinant of J. `detJ[i]` is equal to
    `det(J[i,:,:])`. It must have dimension 1.
K
    The
    inverse of J, `K[i,:,:] = J[i,:,:]^-1`. It must
    have dimension 3.
U
    The input `u` mapped to the reference element. It
    must have dimension 3.
)";

const std::string finite_element__num_entity_dofs
    = R"(Get the number of dofs on each topological entity: (vertices,
edges, faces, cell) in that order. For example, Lagrange degree 2
on a triangle has vertices: [1, 1, 1], edges: [1, 1, 1], cell: [0]
The sum of the entity dofs must match the total number of dofs
reported by FiniteElement::dim,
@code{.cpp}
const std::vector<std::vector<int>>& dofs = e.entity_dofs();
int num_dofs0 = dofs[1][3]; // Number of dofs associated with edge 3
int num_dofs1 = dofs[2][0]; // Number of dofs associated with face 0
@endcode

Returns
=======
Number of dofs associated with an entity of a given
topological dimension. The shape is (tdim + 1, num_entities).
)";

const std::string finite_element__num_entity_closure_dofs
    = R"(Get the number of dofs on the closure of each topological entity:
(vertices, edges, faces, cell) in that order. For example, Lagrange degree
2 on a triangle has vertices: [1, 1, 1], edges: [3, 3, 3], cell: [6]

Returns
=======
Number of dofs associated with the closure of an entity of a given
topological dimension. The shape is (tdim + 1, num_entities).
)";

const std::string finite_element__entity_dofs
    = R"(Get the dofs on each topological entity: (vertices,
edges, faces, cell) in that order. For example, Lagrange degree 2
on a triangle has vertices: [[0], [1], [2]], edges: [[3], [4], [5]],
cell: [[]]

Returns
=======
Dofs associated with an entity of a given
topological dimension. The shape is (tdim + 1, num_entities, num_dofs).
)";

const std::string finite_element__entity_closure_dofs
    = R"(Get the dofs on the closure of each topological entity: (vertices,
edges, faces, cell) in that order. For example, Lagrange degree 2
on a triangle has vertices: [[0], [1], [2]], edges: [[1, 2, 3], [0, 2, 4],
[0, 1, 5]], cell: [[0, 1, 2, 3, 4, 5]]

Returns
=======
Dofs associated with the closure of an entity of a given
topological dimension. The shape is (tdim + 1, num_entities, num_dofs).
)";

const std::string finite_element__base_transformations
    = R"(Get the base transformations
The base transformations represent the effect of rotating or reflecting
a subentity of the cell on the numbering and orientation of the DOFs.
This returns a list of matrices with one matrix for each subentity
permutation in the following order:
Reversing edge 0, reversing edge 1, ...
Rotate face 0, reflect face 0, rotate face 1, reflect face 1, ...

Example: Order 3 Lagrange on a triangle
---------------------------------------
This space has 10 dofs arranged like:
~~~~~~~~~~~~~~~~
2
|\
6 4
|  \
5 9 3
|    \
0-7-8-1
~~~~~~~~~~~~~~~~
For this element, the base transformations are:
[Matrix swapping 3 and 4,
Matrix swapping 5 and 6,
Matrix swapping 7 and 8]
The first row shows the effect of reversing the diagonal edge. The
second row shows the effect of reversing the vertical edge. The third
row shows the effect of reversing the horizontal edge.

Example: Order 1 Raviart-Thomas on a triangle
---------------------------------------------
This space has 3 dofs arranged like:
~~~~~~~~~~~~~~~~
|\
| \
|  \
<-1   0
|  / \
| L ^ \
|   |  \
---2---
~~~~~~~~~~~~~~~~
These DOFs are integrals of normal components over the edges: DOFs 0 and 2
are oriented inward, DOF 1 is oriented outwards.
For this element, the base transformation matrices are:
~~~~~~~~~~~~~~~~
0: [[-1, 0, 0],
[ 0, 1, 0],
[ 0, 0, 1]]
1: [[1,  0, 0],
[0, -1, 0],
[0,  0, 1]]
2: [[1, 0,  0],
[0, 1,  0],
[0, 0, -1]]
~~~~~~~~~~~~~~~~
The first matrix reverses DOF 0 (as this is on the first edge). The second
matrix reverses DOF 1 (as this is on the second edge). The third matrix
reverses DOF 2 (as this is on the third edge).

Example: DOFs on the face of Order 2 Nedelec first kind on a tetrahedron
------------------------------------------------------------------------
On a face of this tetrahedron, this space has two face tangent DOFs:
~~~~~~~~~~~~~~~~
|\        |\
| \       | \
|  \      | ^\
|   \     | | \
| 0->\    | 1  \
|     \   |     \
------    ------
~~~~~~~~~~~~~~~~
For these DOFs, the subblocks of the base transformation matrices are:
~~~~~~~~~~~~~~~~
rotation: [[-1, 1],
[ 1, 0]]
reflection: [[0, 1],
[1, 0]]
~~~~~~~~~~~~~~~~
)";

const std::string finite_element__entity_transformations
    = R"(Return the entity dof transformation matricess
)";

const std::string finite_element__permute_dofs
    = R"(Permute the dof numbering on a cell

Parameters
=========
dofs
    The dof numbering for the cell
cell_info
    The permutation info for the cell
)";

const std::string finite_element__unpermute_dofs
    = R"(Unpermute the dof numbering on a cell

Parameters
=========
dofs
    The dof numbering for the cell
cell_info
    The permutation info for the cell
)";

const std::string finite_element__apply_dof_transformation
    = R"(Apply DOF transformations to some data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string finite_element__apply_transpose_dof_transformation
    = R"(Apply transpose DOF transformations to some data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string finite_element__apply_inverse_transpose_dof_transformation
    = R"(Apply inverse transpose DOF transformations to some data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string finite_element__apply_inverse_dof_transformation
    = R"(Apply inverse DOF transformations to some data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string finite_element__apply_dof_transformation_to_transpose
    = R"(Apply DOF transformations to some transposed data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string
    finite_element__apply_transpose_dof_transformation_to_transpose
    = R"(Apply transpose DOF transformations to some transposed data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string
    finite_element__apply_inverse_transpose_dof_transformation_to_transpose
    = R"(Apply inverse transpose DOF transformations to some transposed data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string finite_element__apply_inverse_dof_transformation_to_transpose
    = R"(Apply inverse DOF transformations to some transposed data

Parameters
=========
data
    The data
block_size
    The number of data points per DOF
cell_info
    The permutation info for the cell
)";

const std::string finite_element__points
    = R"(Return the interpolation points, i.e. the coordinates on the
reference element where a function need to be evaluated in order
to interpolate it in the finite element space.

Returns
=======
Array of coordinate with shape `(num_points, tdim)`
)";

const std::string finite_element__num_points
    = R"(Return the number of interpolation points
)";

const std::string finite_element__interpolation_matrix
    = R"(Return a matrix of weights interpolation
To interpolate a function in this finite element, the functions
should be evaluated at each point given by
FiniteElement::points(). These function values should then be
multiplied by the weight matrix to give the coefficients of the
interpolated function.
)";

const std::string finite_element__interpolate
    = R"(Compute the coefficients of a function given the values of the function
at the interpolation points.

Parameters
=========
coefficients
    The coefficients of the function's
    interpolation into the function space
data
    The function evaluated at the points given by `points()`
block_size
    The block size of the data
)";

const std::string finite_element__map_type = R"(Element map type
)";

const std::string finite_element___map_type
    = R"(The mapping used to map this element from the reference to a cell
)";

const std::string finite_element___matM
    = R"(The interpolation weights and points
)";

const std::string finite_element___matM_new = R"(Interpolation matrices
)";

const std::string finite_element___dof_transformations_are_permutations
    = R"(Indicates whether or not the DOF transformations are all permutations
)";

const std::string finite_element___dof_transformations_are_identity
    = R"(Indicates whether or not the DOF transformations are all identity
)";

const std::string finite_element___eperm
    = R"(The entity permutations (factorised). This will only be set if
_dof_transformations_are_permutations is True and
_dof_transformations_are_identity is False
)";

const std::string finite_element___eperm_rev
    = R"(The reverse entity permutations (factorised). This will only be set if
_dof_transformations_are_permutations is True and
_dof_transformations_are_identity is False
)";

const std::string finite_element___etrans
    = R"(The entity transformations in precomputed form
)";

const std::string finite_element___etransT
    = R"(The transposed entity transformations in precomputed form
)";

const std::string finite_element___etrans_inv
    = R"(The inverse entity transformations in precomputed form
)";

const std::string finite_element___etrans_invT
    = R"(The inverse transpose entity transformations in precomputed form
)";

const std::string finite_element__version
    = R"(Return the version number of basix across projects

Returns
=======
version string
)";

const std::string e_raviart_thomas__create_rt = R"(Create Raviart-Thomas element

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string element_families__lagrange_variant
    = R"(An enum defining the variants of a Lagrange space that can be created
)";

const std::string element_families__family
    = R"(Enum of available element families
)";

const std::string maps__type = R"(Cell type
)";

const std::string maps__apply_map
    = R"(Apply a map to data. Note that the required input arguments depends
on the type of map.

Parameters
=========
u
    The field after mapping, flattened with row-major
    layout
U
    The field to be mapped, flattened with row-major layout
J
    Jacobian of the map
detJ
    Determinant of `J`
K
    The inverse of `J`
map_type
    The map type
)";

const std::string e_nedelec__create_nedelec
    = R"(Create Nedelec element (first kind)

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string e_nedelec__create_nedelec2
    = R"(Create Nedelec element (second kind)

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string lattice__type
    = R"(The type of point spacing to be used in a lattice.

lattice::type::equispaced represents equally spaced points
on an interval and a regularly spaced set of points on other
shapes.

lattice::type::gll represents the GLL (Gauss-Lobatto-Legendre)
points.

lattice::type::chebyshev represents the Chebyshev points.

lattice::type::gl represents the GL (Gauss-Legendre) points.

lattice::type::chebyshev_plus_endpoints represents the Chebyshev points plus
the endpoints of the interval. lattice::type::gl_plus_endpoints represents
the Chebyshev points plus the endpoints of the interval. These points are
only intended for internal use.
)";

const std::string lattice__simplex_method
    = R"(The method used to generate points inside simplices.

lattice::simplex_method::none can be used when no method is needed (eg when
making points on a quadrilateral, or when making equispaced points).

lattice::simplex_method::warp will use the warping defined in Hesthaven and
Warburton, Nodal Discontinuous Galerkin Methods, 2008, pp 175-180
(https://doi.org/10.1007/978-0-387-72067-8).

lattice::simplex_method::isaac will use the method described in Isaac,
Recursive, Parameter-Free, Explicitly Defined Interpolation Nodes for
Simplices, 2020 (https://doi.org/10.1137/20M1321802).

lattice::simplex_method::centroid will place points at the centroids of the
grid created by putting points on the edges, as described in Blyth and
Pozrikidis, A Lobatto interpolation grid over the triangle, 2001
(https://doi.org/10.1093/imamat/hxh077).
)";

const std::string lattice__create
    = R"(Create a lattice of points on a reference cell
optionally including the outer surface points

For a given celltype, this creates a set of points on a regular grid
which covers the cell, e.g. for a quadrilateral, with n=2, the points are:
[0,0],[0.5,0],[1,0],[0,0.5],[0.5,0.5],[1,0.5],[0,1],[0.5,1],[1,1]
If the parameter exterior is set to false, the points lying on the external
boundary are omitted, in this case for a quadrilateral with n=2, the points
are: [0.5,0.5]. The lattice type can be chosen as "equispaced" or
"gll". The "gll" lattice has points spaced along each edge at
the Gauss-Lobatto-Legendre quadrature points. These are the same as
"equispaced" when n<3.

Parameters
=========
celltype
    The cell::type
n
    Size in each direction. There are n+1 points along each edge of the
    cell.
exterior
    If set, includes outer boundaries
type
    A lattice type
simplex_method
    The method used to generate points on simplices

Returns
=======
Set of points
)";

const std::string moments__moments = R"(## Integral moments
These functions generate dual set matrices for integral moments
against spaces on a subentity of the cell
)";

const std::string moments__create_dot_moment_dof_transformations
    = R"(Create the dof transformations for the DOFs defined using a dot
integral moment.

A dot integral moment is defined by
\f[l_i(v) = \int v\cdot\phi_i,\f]
where \f$\phi_i\f$ is a basis function in the moment space, and \f$v\f$ and
\f$\phi_i\f$ are either both scalars or are vectors of the same size.

If the moment space is an interval, this returns one matrix
representing the reversal of the interval. If the moment space is a
face, this returns two matrices: one representing a rotation, the
other a reflection.

These matrices are computed by calculation the interpolation
coefficients of a rotated/reflected basis into the original basis.

Parameters
=========
moment_space
    The finite element space that the integral
    moment is taken against

Returns
=======
A list of dof transformations
)";

const std::string moments__create_moment_dof_transformations
    = R"(Create the DOF transformations for the DOFs defined using an integral
moment.

An integral moment is defined by
\f[l_{i,j}(v) = \int v\cdot e_j\phi_i,\f]
where \f$\phi_i\f$ is a basis function in the moment space, \f$e_j\f$ is a
coordinate direction (of the cell sub-entity the moment is taken on),
\f$v\f$ is a vector, and \f$\phi_i\f$ is a scalar.

This will combine multiple copies of the result of
`create_dot_moment_dof_transformations` to give the transformations
for integral moments of each vector component against the moment
space.

Parameters
=========
moment_space
    The finite element space that the integral
    moment is taken against

Returns
=======
A list of dof transformations
)";

const std::string moments__create_normal_moment_dof_transformations
    = R"(Create the dof transformations for the DOFs defined using a normal
integral moment.

A normal integral moment is defined by
\f[l_{i,j}(v) = \int v\cdot n\phi_i,\f]
where \f$\phi_i\f$ is a basis function in the moment space, \f$n\f$ is
normal to the cell sub-entity, \f$v\f$ is a vector, and \f$\phi_i\f$ is a
scalar.

This does the same as `create_dot_moment_dof_transformations` with
some additional factors of -1 to account for the changing of the
normal direction when the entity is reflected.

Parameters
=========
moment_space
    The finite element space that the integral
    moment is taken against

Returns
=======
A list of dof transformations
)";

const std::string moments__create_tangent_moment_dof_transformations
    = R"(Create the dof transformations for the DOFs defined using a
tangential integral moment.

A tangential integral moment is defined by
\f[l_{i,j}(v) = \int v\cdot t\phi_i,\f]
where \f$\phi_i\f$ is a basis function in the moment space, \f$t\f$ is
tangential to the edge, \f$v\f$ is a vector, and \f$\phi_i\f$ is a scalar.

This does the same as `create_dot_moment_dof_transformations` with
some additional factors of -1 to account for the changing of the
tangent direction when the edge is reflected.

Parameters
=========
moment_space
    The finite element space that the integral
    moment is taken against

Returns
=======
A list of dof transformations
)";

const std::string moments__make_integral_moments
    = R"(Make interpolation points and weights for simple integral moments

These will represent the integral of each function in the moment
space over each sub entity of the moment space's cell type in a cell
with the given type. For example, if the input cell type is a
triangle, and the moment space is a P1 space on an edge, this will
perform two integrals for each of the 3 edges of the triangle.

Parameters
=========
moment_space
    The space to compute the integral moments against
celltype
    The cell type of the cell on which the space is
    being defined
value_size
    The value size of the space being defined
q_deg
    The quadrature degree used for the integrals
)";

const std::string moments__make_dot_integral_moments
    = R"(Make interpolation points and weights for dot product integral
moments

These will represent the integral of each function in the moment
space over each sub entity of the moment space's cell type in a cell
with the given type. For example, if the input cell type is a
triangle and the moment space is a P1 space on an edge, this will
perform two integrals for each of the 3 edges of the triangle.

TODO: Clarify what happens value size of the moment space is less
than `value_size`.

Parameters
=========
V
    The space to compute the integral moments against
celltype
    The cell type of the cell on which the space is being
    defined
value_size
    The value size of the space being defined
q_deg
    The quadrature degree used for the integrals
)";

const std::string moments__make_tangent_integral_moments
    = R"(Make interpolation points and weights for tangent integral moments

These can only be used when the moment space is defined on edges of
the cell

Parameters
=========
V
    The space to compute the integral moments against
celltype
    The cell type of the cell on which the space is
    being defined
value_size
    The value size of the space being defined the
    space
q_deg
    The quadrature degree used for the integrals
)";

const std::string moments__make_normal_integral_moments
    = R"(Compute interpolation points and weights for normal integral moments

These can only be used when the moment space is defined on facets of
the cell

Parameters
=========
V
    The space to compute the integral moments against
celltype
    The cell type of the cell on which the space is
    being defined
value_size
    The value size of the space being defined
q_deg
    The quadrature degree used for the integrals

Returns
=======
(interpolation points, interpolation matrix)
)";

const std::string e_bubble__create_bubble
    = R"(Create a bubble element on cell with given degree

Parameters
=========
celltype
    interval, triangle, tetrahedral, quadrilateral or
    hexahedral celltype
degree
    TODO: document this
discontinuous
    TODO: document this

Returns
=======
A FiniteElement
)";

const std::string precompute__prepare_permutation = R"(Prepare a permutation

This computes a representation of the permutation that allows the
permutation to be applied without any temporary memory assignment.

In pseudo code, this function does the following:

\code{.pseudo}
FOR index, entry IN perm:
new_index = entry
WHILE new_index < index:
new_index = perm[new_index]
OUT[index] = new_index
\endcode

Example
-------
As an example, consider the permutation `P = [1, 4, 0, 5, 2, 3]`.

First, we look at the 0th entry. `P[0]` is 1. This is greater than 0, so the
0th entry of the output is 1.

Next, we look at the 1st entry. `P[1]` is 4. This is greater than 1, so the
1st entry of the output is 4.

Next, we look at the 2nd entry. `P[2]` is 0. This is less than 2, so we look
at `P[0]. `P[0]` is 1. This is less than 2, so we look at `P[1]`. `P[1]`
is 4. This is greater than 2, so the 2nd entry of the output is 4.

Next, we look at the 3rd entry. `P[3]` is 5. This is greater than 3, so the
3rd entry of the output is 5.

Next, we look at the 4th entry. `P[4]` is 2. This is less than 4, so we look
at `P[2]`. `P[2]` is 0. This is less than 4, so we look at `P[0]`. `P[0]`
is 1. This is less than 4, so we look at `P[1]`. `P[1]` is 4. This is
greater than (or equal to) 4, so the 4th entry of the output is 4.

Next, we look at the 5th entry. `P[5]` is 3. This is less than 5, so we look
at `P[3]`. `P[3]` is 5. This is greater than (or equal to) 5, so the 5th
entry of the output is 5.

Hence, the output of this function in this case is `[1, 4, 4, 5, 4, 5]`.

For an example of how the permutation in this form is applied, see
`apply_permutation()`.

Parameters
=========
perm
    A permutation

Returns
=======
The precomputed representation of the permutation
)";

const std::string precompute__apply_permutation
    = R"(Apply a (precomputed) permutation

This uses the representation returned by `prepare_permutation()` to apply a
permutation without needing any temporary memory.

In pseudo code, this function does the following:

\code{.pseudo}
FOR index, entry IN perm:
SWAP(data[index], data[entry])
\endcode

If `block_size` is set, this will apply the permutation to every block.
The `offset` is set, this will start applying the permutation at the
`offset`th block.

Example
-------
As an example, consider the permutation `P = [1, 4, 0, 5, 2, 3]`.
In the documentation of `prepare_permutation()`, we saw that the precomputed
representation of this permutation is `P2 = [1, 4, 4, 5, 4, 5]`. In this
example, we look at how this representation can be used to apply this
permutation to the array `A = [a, b, c, d, e, f]`.

`P2[0]` is 1, so we swap `A[0]` and `A[1]`. After this, `A = [b, a, c, d, e,
f]`.

`P2[1]` is 4, so we swap `A[1]` and `A[4]`. After this, `A = [b, e, c, d, a,
f]`.

`P2[2]` is 4, so we swap `A[2]` and `A[4]`. After this, `A = [b, e, a, d, c,
f]`.

`P2[3]` is 5, so we swap `A[3]` and `A[5]`. After this, `A = [b, e, a, f, c,
d]`.

`P2[4]` is 4, so we swap `A[4]` and `A[4]`. This changes nothing.

`P2[5]` is 5, so we swap `A[5]` and `A[5]`. This changes nothing.

Therefore the result of applying this permutation is `[b, e, a, f, c, d]`
(which is what we get if we apply the permutation directly).

Parameters
=========
perm
    A permutation in precomputed form (as returned by
    `prepare_permutation()`)
data
    The data to apply the permutation to
offset
    The position in the data to start applying the permutation
block_size
    The block size of the data
)";

const std::string precompute__apply_permutation_to_transpose
    = R"(Apply a (precomputed) permutation to some transposed data

see `apply_permutation()`.
)";

const std::string precompute__prepare_matrix = R"(Prepare a matrix

This computes a representation of the matrix that allows the matrix to be
applied without any temporary memory assignment.

This function will first permute the matrix's columns so that the top left
$n\times n$ blocks are invertible (for all $n$). Let $A$ be the
input matrix after the permutation is applied. The output vector $D$ and
matrix $M$ are then given by:
@f{align*}{
D_i &= \begin{cases}
A_{i, i} & i = 0\\
A_{i, i} - A_{i,:i}A_{:i,:i}^{-1}A_{:i,i} & i \not= 0
\end{cases},\\
M_{i,j} &= \begin{cases}
A_{i,:i}A_{:i,:i}^{-1}e_j & j < i\\
0 & j = i\\
A_{i, i} - A_{i,:i}A_{:i,:i}^{-1}A_{:i,j} & j > i = 0
\end{cases},
@f}
where $e_j$ is the $j$th coordinate vector, we index all the
matrices and vector starting at 0, and we use numpy-slicing-stying notation
in the subscripts: for example, $A_{:i,j}$ represents the first $i$
entries in the $j$th column of $A$

This function returns the permutation (precomputed as in
`prepare_permutation()`), the vector $D$, and the matrix $M$ as a
tuple.

Example
-------
As an example, consider the matrix $A = $ `[[-1, 0, 1], [1, 1, 0], [2,
0, 2]]`. For this matrix, no permutation is needed, so the first item in the
output will represent the identity permutation. We now compute the output
vector $D$ and matrix $M$.

First, we set $D_0 = A_{0,0}=-1$,
set the diagonal of $M$ to be 0
and set $M_{0, 1:} = A_{0, 1:}=\begin{bmatrix}0&1\end{bmatrix}$.
The output so far is
@f{align*}{ D &= \begin{bmatrix}-1\\?\\?\end{bmatrix},\\
\quad M &= \begin{bmatrix}
0&0&1\\
?&0&?\\
?&?&0
\end{bmatrix}. @f}

Next, we set:
@f{align*}{ D_1 &= A_{1,1} - A_{1, :1}A_{:1,:1}^{-1}A_{:1, 1}\\
&= 1 -
\begin{bmatrix}-1\end{bmatrix}\cdot\begin{bmatrix}0\end{bmatrix}\\
&= 1,\\
M_{2,0} &= A_{1, :1}A_{:1,:1}^{-1}e_0\\
&= \begin{bmatrix}1\end{bmatrix}\begin{bmatrix}-1\end{bmatrix}^{-1}
\begin{bmatrix}1\end{bmatrix}\\
&= \begin{bmatrix}-1\end{bmatrix}
M_{2,3} &= A_{1,2}-A_{1, :1}A_{:1,:1}^{-1}A_{:1, 1}\\
&=
0-\begin{bmatrix}1\end{bmatrix}\begin{bmatrix}-1\end{bmatrix}^{-1}
\begin{bmatrix}1\end{bmatrix},\\
&= 1.
@f}
The output so far is
@f{align*}{ D &= \begin{bmatrix}-1\\1\\?\end{bmatrix},\\
\quad M &= \begin{bmatrix}
0&0&1\\
-1&0&1\\
?&?&0
\end{bmatrix}. @f}

Next, we set:
@f{align*}{ D_2 &= A_{2,2} - A_{2, :2}A_{:2,:2}^{-1}A_{:2, 2}\\
&= 2 -
\begin{bmatrix}2&0\end{bmatrix}
\begin{bmatrix}-1&0\\1&1\end{bmatrix}^{-1}
\begin{bmatrix}1\\0\end{bmatrix}\\
&= 4,\\
M_{2,0} &= A_{2, :2}A_{:2,:2}^{-1}e_0\\ &= -2.\\
M_{2,1} &= A_{2, :2}A_{:2,:2}^{-1}e_1\\ &= 0.\\
@f}
The output is
@f{align*}{ D &= \begin{bmatrix}-1\\1\\4\end{bmatrix},\\
\quad M &= \begin{bmatrix}
0&0&1\\
-1&0&1\\
-2&0&0
\end{bmatrix}. @f}

For an example of how the permutation in this form is applied, see
`apply_matrix()`.

Parameters
=========
matrix
    A matrix

Returns
=======
The precomputed representation of the matrix
)";

const std::string precompute__apply_matrix = R"(Apply a (precomputed) matrix

This uses the representation returned by `prepare_matrix()` to apply a
matrix without needing any temporary memory.

In pseudo code, this function does the following:

\code{.pseudo}
perm, diag, mat = matrix
apply_permutation(perm, data)
FOR index IN RANGE(dim):
data[index] *= diag[index]
FOR j IN RANGE(dim):
data[index] *= mat[index, j] * data[j]
\endcode

If `block_size` is set, this will apply the permutation to every block.
The `offset` is set, this will start applying the permutation at the
`offset`th block.

Example
-------
As an example, consider the matrix $A = $ `[[-1, 0, 1], [1, 1, 0], [2,
0, 2]]`. In the documentation of `prepare_matrix()`, we saw that the
precomputed representation of this matrix is the identity permutation,
@f{align*}{ D &= \begin{bmatrix}-1\\1\\4\end{bmatrix},\\
\quad M &= \begin{bmatrix}
0&0&1\\
-1&0&1\\
-2&0&0
\end{bmatrix}. @f}
In this example, we look at how this representation can be used to
apply this matrix to the vector $v = $ `[3, -1, 2]`.

No permutation is necessary, so first, we multiply $v_0$ by
$D_0=-1$. After this, $v$ is `[-3, -1, 2]`.

Next, we add $M_{0,i}v_i$ to $v_0$ for all $i$: in this case, we
add $0\times-3 + 0\times-1 + 1\times2 = 2$. After this, $v$ is `[-1,
-1, 2]`.

Next, we multiply $v_1$ by $D_1=1$. After this, $v$ is `[-1, -1,
2]`.

Next, we add $M_{1,i}v_i$ to $v_1$ for all $i$: in this case, we
add $-1\times-1 + 0\times-1 + 1\times2 = 3$. After this, $v$ is
`[-1, 2, 2]`.

Next, we multiply $v_2$ by $D_2=4$. After this, $v$ is `[-1, 2,
8]`.

Next, we add $M_{2,i}v_i$ to $v_2$ for all $i$: in this case, we
add $-2\times-1 + 0\times2 + 0\times8 = 2$. After this, $v$ is `[-1,
2, 10]`. This final value of $v$ is what the result of $Av$

Parameters
=========
matrix
    A matrix in precomputed form (as returned by
    `prepare_matrix()`)
data
    The data to apply the permutation to
offset
    The position in the data to start applying the permutation
block_size
    The block size of the data
)";

const std::string precompute__apply_matrix_to_transpose
    = R"(Apply a (precomputed) matrix to some transposed data.

See `apply_matrix()`.
)";

const std::string polyset__tabulate
    = R"(Tabulate the orthonormal polynomial basis, and derivatives, at
points on the reference cell.

All derivatives up to the given order are computed. If derivatives
are not required, use `n = 0`. For example, order `n = 2` for a 2D
cell, will compute the basis \f$\psi, d\psi/dx, d\psi/dy, d^2
\psi/dx^2, d^2\psi/dxdy, d^2\psi/dy^2\f$ in that order (0, 0), (1,
0), (0, 1), (2, 0), (1, 1), (0 ,2).

For an interval cell there are `nderiv + 1` derivatives, for a 2D
cell, there are `(nderiv + 1)(nderiv + 2)/2` derivatives, and in 3D,
there are `(nderiv + 1)(nderiv + 2)(nderiv + 3)/6`. The ordering is
'triangular' with the lower derivatives appearing first.

Parameters
=========
celltype
    Cell type
d
    Polynomial degree
n
    Maximum derivative order. Use n = 0 for the basis only.
x
    Points at which to evaluate the basis. The shape is
    (number of points, geometric dimension).

Returns
=======
Polynomial sets, for each derivative, tabulated at points.
The shape is `(number of derivatives computed, number of points,
basis index)`.

- The first index is the derivative. The first entry is the basis
itself. Derivatives are stored in triangular (2D) or tetrahedral
(3D) ordering, e.g. if `(p, q)` denotes `p` order dervative with
repsect to `x` and `q` order derivative with respect to `y`, [0] ->
(0, 0), [1] -> (1, 0), [2] -> (0, 1), [3] -> (2, 0), [4] -> (1, 1),
[5] -> (0, 2), [6] -> (3, 0),...
The function basix::idx maps tuples `(p, q, r)` to the array index.

- The second index is the point, with index `i` correspondign to the
point in row `i` of @p x.

- The third index is the basis function index.
TODO: Does the order for the third index need to be documented?
)";

const std::string polyset__dim = R"(Dimension of a polynomial space

Parameters
=========
cell
    The cell type
d
    The polynomial degree

Returns
=======
The number terms in the basis spanning a space of
polynomial degree @p d
)";

const std::string e_serendipity__create_serendipity
    = R"(Create a serendipity element on cell with given degree

Parameters
=========
celltype
    quadrilateral or hexahedral celltype
degree
    TODO: document this
discontinuous
    TODO: document this

Returns
=======
A FiniteElement
)";

const std::string e_serendipity__create_serendipity_div
    = R"(Create a serendipity H(div) element on cell with given degree

Parameters
=========
celltype
    quadrilateral or hexahedral celltype
degree
    TODO: document this
discontinuous
    TODO: document this

Returns
=======
A FiniteElement
)";

const std::string e_serendipity__create_serendipity_curl
    = R"(Create a serendipity H(curl) element on cell with given degree

Parameters
=========
celltype
    quadrilateral or hexahedral celltype
degree
    TODO: document this
discontinuous
    TODO: document this

Returns
=======
A FiniteElement
)";

const std::string quadrature__compute_jacobi_deriv
    = R"(Evaluate the nth Jacobi polynomial and derivatives with weight
parameters (a, 0) at points x

Parameters
=========
a
    Jacobi weight a
n
    Order of polynomial
nderiv
    Number of derivatives (if zero, just compute
    polynomial itself)
x
    Points at which to evaluate

Returns
=======
s Array of polynomial derivative values (rows) at points
(columns)
)";

const std::string quadrature__compute_gauss_jacobi_points
    = R"(Finds the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's method.

Parameters
=========
a
    weight in Jacobi (b=0)
m
    order

Returns
=======
list of points in 1D
)";

const std::string quadrature__compute_gauss_jacobi_rule
    = R"(Gauss-Jacobi quadrature rule (points and weights)
)";

const std::string quadrature__make_quadrature_line
    = R"(Compute line quadrature rule on [0, 1]

Parameters
=========
m
    order

Returns
=======
s list of points, list of weights
)";

const std::string quadrature__make_quadrature_triangle_collapsed
    = R"(Compute triangle quadrature rule on [0, 1]x[0, 1]

Parameters
=========
m
    order

Returns
=======
s list of points, list of weights
)";

const std::string quadrature__make_quadrature_tetrahedron_collapsed
    = R"(Compute tetrahedron quadrature rule on [0, 1]x[0, 1]x[0, 1]

Parameters
=========
m
    order

Returns
=======
s List of points, list of weights. The number of points
arrays has shape (num points, gdim)
)";

const std::string quadrature__make_quadrature
    = R"(Utility for quadrature rule on reference cell

Parameters
=========
rule
    Name of quadrature rule (or use "default")
celltype
    TODO: document this
m
    Maximum degree of polynomial that this quadrature rule
    will integrate exactly

Returns
=======
s List of points and list of weights. The number of points
arrays has shape (num points, gdim)
)";

const std::string quadrature__make_gll_line
    = R"(Compute GLL line quadrature rule on [0, 1]

Parameters
=========
m
    order

Returns
=======
s list of 1D points, list of weights
)";

const std::string quadrature__compute_gll_rule
    = R"(GLL quadrature rule (points and weights)
)";

const std::string cell__type = R"(Cell type
)";

const std::string cell__geometry = R"(Cell geometry

Parameters
=========
celltype
    Cell Type

Returns
=======
Set of vertex points of the cell
)";

const std::string cell__topology = R"(Cell topology

Parameters
=========
celltype
    Cell Type

Returns
=======
List of topology (vertex indices) for each dimension (0..tdim)
)";

const std::string cell__sub_entity_connectivity
    = R"(Get the numbers of entities connected to each subentity of the cell.

Returns a vector of the form: output[dim][entity_n][connected_dim] =
[connected_entity_n0, connected_entity_n1, ...] This indicates that the
entity of dimension `dim` and number `entity_n` is connected to the entities
of dimension `connected_dim` and numbers `connected_entity_n0`,
`connected_entity_n1`, ...

Parameters
=========
celltype
    Cell Type

Returns
=======
List of topology (vertex indices) for each dimension (0..tdim)
)";

const std::string cell__sub_entity_geometry
    = R"(Sub-entity of a cell, given by topological dimension and index

Parameters
=========
celltype
    The cell::type
dim
    Dimension of sub-entity
index
    Local index of sub-entity

Returns
=======
Set of vertex points of the sub-entity
)";

const std::string cell__num_sub_entities = R"(TODO: Optimise this function
Number of sub-entities of a cell by topological dimension

Parameters
=========
celltype
    The cell::type
dim
    Dimension of sub-entity

Returns
=======
The number of sub-entities of the given dimension
@warning This function is expensive to call. Do not use in
performance critical code
)";

const std::string cell__topological_dimension
    = R"(Get the topological dimension for a given cell type

Parameters
=========
celltype
    Cell type

Returns
=======
the topological dimension
)";

const std::string cell__sub_entity_type
    = R"(Get the cell type of a sub-entity of given dimension and index

Parameters
=========
celltype
    Type of cell
dim
    Topological dimension of sub-entity
index
    Index of sub-entity

Returns
=======
cell type of sub-entity
)";

const std::string cell__volume = R"(Get the volume of a reference cell

Parameters
=========
cell_type
    Type of cell
)";

const std::string cell__facet_outward_normals
    = R"(Get the (outward) normals to the facets of a reference cell

Parameters
=========
cell_type
    Type of cell
)";

const std::string cell__facet_normals
    = R"(Get the normals to the facets of a reference cell oriented using the
low-to-high ordering of the facet

Parameters
=========
cell_type
    Type of cell
)";

const std::string cell__facet_orientations
    = R"(Get a array of bools indicating whether or not the facet normals are outward
pointing

Parameters
=========
cell_type
    Type of cell
)";

const std::string cell__facet_reference_volumes
    = R"(Get the reference volumes of the facets of a reference cell

Parameters
=========
cell_type
    Type of cell
)";

const std::string cell__subentity_types
    = R"(Get the reference volumes of the facets of a reference cell

Parameters
=========
cell_type
    Type of cell
)";

const std::string cell__facet_jacobians
    = R"(Get the jacobians of the facets of a reference cell

Parameters
=========
cell_type
    Type of cell
)";

const std::string e_nce_rtc__create_rtc = R"(Create RTC H(div) element

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string e_nce_rtc__create_nce = R"(Create NC H(curl) element

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string e_regge__create_regge = R"(Create Regge element

Parameters
=========
celltype
    TODO: document this
degree
    TODO: document this
discontinuous
    TODO: document this
)";

const std::string interpolation__compute_interpolation_operator
    = R"(Computes a matrix that represents the interpolation between two
elements.

If the two elements have the same value size, this function returns
the interpolation between them.

If element_from has value size 1 and element_to has value size > 1, then
this function returns a matrix to interpolate from a blocked element_from
(ie multiple copies of element_from) into element_to.

If element_to has value size 1 and element_from has value size > 1, then
this function returns a matrix that interpolates the components of
element_from into copies of element_to.

NOTE: If the elements have different value sizes and both are
greater than 1, this function throws a runtime error

In order to interpolate functions between finite element spaces on arbitrary
cells, the functions must be pulled back to the reference element (this pull
back includes applying DOF transformations). The matrix that this function
returns can then be applied, then the result pushed forward to the cell. If
element_from and element_to have the same map type, then only the DOF
transformations need to be applied, as the pull back and push forward cancel
each other out.

Parameters
=========
element_from
    The element to interpolate from
element_to
    The element to interpolate to

Returns
=======
Matrix operator that maps the 'from' degrees-of-freedom to
the 'to' degrees-of-freedom
)";

} // namespace basix::docstring
