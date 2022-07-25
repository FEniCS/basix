# =========================
# Creating a custom element
# =========================
#
# In Basix, it is possible to create custom finite elements. This demo describes
# how to do this and what data you need to provide when creating a custom element.
#
# First, we import Basix and Numpy.

import basix
import numpy as np
from basix import CellType, MapType, PolynomialType, LatticeType

# Lagrange element with bubble
# ============================
#
# As a first example, we create a degree 1 Lagrange element with a quadratic bubble
# on a quadrilateral cell. This element will span the following set of polynomials:
#
# .. math::
#    \left\{1,\; y,\; x,\; xy,\; x(1-x)y(1-y)\right\}.
#
# We will define the degrees of freedom (DOFs) of this element by placing a point
# evaluation at each vertex, plus one at the midpoint of the cell.
#
# Polynomial coefficients
# -----------------------
#
# When creating a custom element, we must input the coefficients that define
# a basis of the set of polynomials that our element spans. In this example,
# we will represent the 5 functions above in terms of the 9 orthogonal polynomials
# of degree :math:`\leqslant2` on a quadrilateral, so we create a 5 by 9 matrix.

wcoeffs = np.zeros((5, 9))

# The degree 2 orthonormal polynomials for a quadrilateral will have their highest
# degree terms in the following order:
#
# .. math::
#   1,\; y,\; y^2,\; x,\; xy,\; xy^2,\; x^2,\; x^2y,\; x^2y^2
#
# The order in which the polynomials appear in the orthonormal polynomial sets for
# each cell are documented at
# https://docs.fenicsproject.org/basix/main/polyset-order.html.
#
# As our polynomial space contains 1, :math:`y`, :math:`x` and :math:`xy`. The first
# four rows of the matrix contain a single 1 for the four orthogonal polynomials with
# these are their highest degree terms.

wcoeffs[0, 0] = 1
wcoeffs[1, 1] = 1
wcoeffs[2, 3] = 1
wcoeffs[3, 4] = 1

# The final row of the matrix defines the polynomials :math:`x(1-x)y(1-y)`. As the polynomials
# are orthonormal, we can represent this as
#
# .. math::
#    x(1-x)y(1-y) = \sum_{i=0}^8\int_0^1\int_0^1p_i(x, y)x(1-x)y(1-y)\,\mathrm{d}x\,\mathrm{d}y\; p_i(x, y),
#
# where :math:`p_0` to :math:`p_8` are the orthonormal polynomials. Therefore the coefficients we want
# to put in the final row of our matrix are:
#
# .. math::
#    \int_0^1\int_0^1p_i(x, y)x(1-x)y(1-y)\,\mathrm{d}x\,\mathrm{d}y.
#
# We compute these integrals using a degree 4 quadrature rule (this is the largest degree
# that the integrand will be, so these integrals will be exact).

pts, wts = basix.make_quadrature(CellType.quadrilateral, 4)
poly = basix.tabulate_polynomials(PolynomialType.legendre, CellType.quadrilateral, 2, pts)
x = pts[:, 0]
y = pts[:, 1]
f = x * (1 - x) * y * (1 - y)
for i in range(9):
    wcoeffs[4, i] = sum(f * poly[i, :] * wts)

# Interpolation
# -------------
#
# Next, we compute the points and matrices that define how functions can be interpolated
# into this space. These are representations of the functionals that are used in the
# Ciarlet definition of the finite element -- in this example, these are evaluations
# at the points described above.
#
# First, we define the points. We create an array of points for each entity of each
# dimension. For each vertex of the cell, we include the coordinates of that vertex.
# For the interior of the cell, we include a point at :math:`(0.5,0.5)`.
#
# The shape of each of the point lists is (number of points, dimension).

x = [[], [], [], []]
x[0].append(np.array([[0.0, 0.0]]))
x[0].append(np.array([[1.0, 0.0]]))
x[0].append(np.array([[0.0, 1.0]]))
x[0].append(np.array([[1.0, 1.0]]))
x[2].append(np.array([[0.5, 0.5]]))

# There are no DOFs associates with the edges for this element, so we add an empty
# array of points for each edge.

for _ in range(4):
    x[1].append(np.zeros((0, 2)))

# We then define the interpolation matrices that define how the evaluations at the points
# are combined to evaluate the functionals. As all the DOFs are point evaluations in this
# example, the matrices are all identity matrices for the entities that have a point.
#
# The shape of each matrix is (number of DOFs, value size, number of points, number of
# derivatives).

M = [[], [], [], []]
for _ in range(4):
    M[0].append(np.array([[[[1.]]]]))
M[2].append(np.array([[[[1.]]]]))

# There are no DOFs associates with the edges for this element, so we add an empty
# matrix for each edge.

for _ in range(4):
    M[1].append(np.zeros((0, 1, 0, 1)))

# Creating the element
# --------------------
#
# We now create the custom element. The inputs into `basix.create_custom_element` are:
#
# - The cell type. In this example, this is a quadrilateral.
# - The value shape of the element. In this example, this is `[]` as the element is scalar.
# - The coefficients that define the polynomial set. In this example, this is `wcoeffs`.
# - The points used to define interpolation into the element. In this example, this is `x`.
# - The matrix used to define interpolation into the element. In this example, this is `M`.
# - The number of derivates used in the evalutation of the functionals. In this example, this
#   is 0.
# - The map type. In this example, this is the identity map.
# - A bool indicating whether the element is discontinuous. In this example, this is `False`.
# - The highest degree :math:`n` such that all degree :math:`n` polynomials are contained in
#   this set. In this example, this is 1.
# - The highest degree of a polynomial in the element. In this example, this is 2. It is
#   important that this value is correct, as it will be used to determine the number of
#   polynomials to use when creating and tabulating the element.

element = basix.create_custom_element(
    CellType.quadrilateral, [], wcoeffs, x, M, 0, MapType.identity, False, 1, 2)

# We can now use this element in the same way we can use a built-in element. For example, we
# can tabulate the element at a set of points. If the points we use are the same as the points
# we used to define the DOFs, we see that each basis function is equal to 1 at one of these points,
# and equal to zero at all the other points.

points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
print(element.tabulate(0, points))

# Degree 1 Ravairt--Thomas element
# ================================
#
# As a second example, we create a degree 1 Raviart--Thomas element on a triangle. Details of
# the definition of this element can be found at
# https://defelement.com/elements/raviart-thomas.html. This element
# spans:
#
# .. math::
#    \left\{(1, 0),\; (0, 1),\; (x, y)\right\}.
#
# The DOFs of this element are integrals along each edge of the cell of the dot product of the
# function with the normal to the edge.
#
# In contrast to the scalar-valued element above, this element has a value size of 2.
#
# Polynomial coefficients
# -----------------------
#
# In this example, we will represent the 3 functions above in terms of the 3 orthogonal
# polynomials of degree :math:`\leqslant1` on a triangle in each of the two coordinate directions.
# We therefore create a 3 by 6 matrix.

wcoeffs = np.zeros((3, 6))

# The highest degree terms in each polynomial will be:
#
# .. math::
#   (1, 0),\; (x, 0),\; (y, 0),\; (0, 1),\; (0, x),\; (0, y)
#
# We include :math:`(1,0)` and :math:`(0,1)` as the first two rows of the matrix, and use
# integrals to represent :math:`(x,y)` as in the previous example.

wcoeffs[0, 0] = 1
wcoeffs[1, 3] = 1

pts, wts = basix.make_quadrature(CellType.triangle, 2)
poly = basix.tabulate_polynomials(PolynomialType.legendre, CellType.triangle, 1, pts)
x = pts[:, 0]
y = pts[:, 1]
for i in range(3):
    wcoeffs[2, i] = sum(x * poly[i, :] * wts)
    wcoeffs[2, 3 + i] = sum(y * poly[i, :] * wts)

# Interpolation
# -------------
#
# For this element, there will be multiple points used per DOF, as the functionals that define
# the element are integrals. We begin by defining a degree 1 quadrature rule on an interval.
# This quadrature rule will be used to integrate on the edges of the triangle.

pts, wts = basix.make_quadrature(CellType.interval, 1)

# The points associated with each edge are calculated by mapping the quadrature points to each edge.

x = [[], [], [], []]
for _ in range(3):
    x[0].append(np.zeros((0, 2)))
x[1].append(np.array([[1 - p[0], p[0]] for p in pts]))
x[1].append(np.array([[0, p[0]] for p in pts]))
x[1].append(np.array([[p[0], 0] for p in pts]))
x[2].append(np.zeros((0, 2)))

# The interpolation matrices for the edges in this example will be have shape (1, 2, len(pts), 1),
# as there is one DOF per edge, the value size is 2, and we have len(pts) quadrature points on each
# edge, and no extra derivatives are used. The entries of these matrices are the quadrature weights
# multiplied by the normal directions.

M = [[], [], [], []]
for _ in range(3):
    M[0].append(np.zeros((0, 2, 0, 1)))
for normal in [[-1, -1], [-1, 0], [0, 1]]:
    mat = np.empty((1, 2, len(wts), 1))
    mat[0, 0, :, 0] = normal[0] * wts
    mat[0, 1, :, 0] = normal[1] * wts
    M[1].append(mat)
M[2].append(np.zeros((0, 2, 0, 1)))

# Creating the element
# --------------------

element = basix.create_custom_element(
    CellType.triangle, [2], wcoeffs, x, M, 0, MapType.contravariantPiola, False, 0, 1)

# To confirm that we have defined this element correctly, we compare it to the built-in
# Raviart--Thomas element.

rt = basix.create_element(basix.ElementFamily.RT, CellType.triangle, 1)

points = basix.create_lattice(CellType.triangle, 1, LatticeType.equispaced, True)
assert np.allclose(rt.tabulate(0, points), element.tabulate(0, points))
