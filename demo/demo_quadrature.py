# ====================================
# Creating and using a quadrature rule
# ====================================
#
# This demo shows how quadrature rules can be obtained from Basix and how these
# can be used to compute the integrals of functions. A quadrature rule uses a
# set of points (:math:`p_0` to :math:`p_{n-1}`) and weights (:math:`w_0` to
# :math:`w_{n-1}`), and approximates an integral as the weighted sum of the
# values of the function at these points, ie
#
# .. math::
#
#    \int f\,\mathrm{d}x \approx \sum_i w_if(p_i).
#
# First, we import Basix and Numpy.

import basix
import numpy as np
from basix import ElementFamily, CellType, LagrangeVariant

# To get a quadrature rule on a triangle, we use the function `make_quadrature`.
# This function takes two or three three inputs. We want to use the default
# quadrature, pass in two inputs: a cell type and an order. The order of the rule
# is equal to the degree of the highest degree polynomial that will be exactly
# integrated by this rule. In this example, we use an order 4 rule, so all quartic
# polynomials will be integrated exactly.
#
# `make_quadrature` returns two values: the points and the weights of the
# quadrature rule.

points, weights = basix.make_quadrature(CellType.triangle, 4)

# If we want to control the type of quadrature used, we can pass in three
# inputs to `make_quadrautre`. For example, the following code would force basix
# to use a Gauss-Jacobi quadrature rule:

points, weights = basix.make_quadrature(
    basix.QuadratureType.gauss_jacobi, CellType.triangle, 4)

# We now use this quadrature rule to integrate the functions :math:`f(x,y)=x^3y`
# and :math:`g(x,y)=x^3y^2` over the triangle. The exact values of these integrals
# are 1/120 (0.00833333333333...) and 1/420 (0.00238095238095...) respectively.
#
# As :math:`f` is a degree 4 polynomial, we expect our quadrature rule to be able
# to compute its integral exactly (within machine precision). :math:`g` on the
# other hand is a degree 5 polynomial, so its integral will not be computed exactly.
#
# We define Python functions that compute :math:`f` and :math:`g` for every point.
# These functions use features of Numpy to compute all the values at once.


def f(points):
    return points[:, 0] ** 3 * points[:, 1]


def g(points):
    return points[:, 0] ** 3 * points[:, 1] ** 2


# We can now use Numpy features to compute the integrals.

print(np.sum(weights * f(points)))
print(np.sum(weights * g(points)))

# We obtain the values 0.00833333333333334 and 0.002393509368731209. As expected,
# the integral of :math:`f` has been computed to within machine precision, while
# the integral of :math:`g` is correct to 4 decimal places, but is not exact.
#
# Integrating a basis function
# ============================
#
# We next use the quadrature rule to compute the integral of a basis function in
# a degree 3 Lagrange space. We first create the space and tabulate its basis
# functions at the quadrature points.

lagrange = basix.create_element(
    ElementFamily.P, CellType.triangle, 3, LagrangeVariant.equispaced)

values = lagrange.tabulate(0, points)

# We compute the integral of the third (note that the counting starts at 0) basis
# function in this space. We can obtain the values of this basis function from
# `values` by using the indices `[0, :, 3, 0]`. The integral can therefore
# computed as follows. As this basis function will be degree three, the result
# will again be exact (withing machine precision).

print(np.sum(weights * values[0, :, 3, 0]))
