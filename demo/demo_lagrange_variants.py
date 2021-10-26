# =============================
# Variants of Lagrange elements
# =============================
#
# When creating high degree function spaces, it is important to define the
# degrees of freedom (DOFs) of a space in a way that leads to well behaved
# basis functions. For example, if a Lagrange space is created using equally
# spaced points, then the basis functions with exhibit Runge's phenomenon
# and large peaks will be observed near the edges of the cell.
#
# When a finite element is defined using points evaluation DOFs, we can use
# the Lebesgue constant to indicate how well behaved a set of basis functions
# will be. The Lebesgue constant is defined by
#
# .. math::
#    \Lambda = \max_{x\in R}\left(\sum_i\left|\phi_i(x)\right|\right),
#
# where :math:`R` is the reference cell and :math:`\phi_0` to :math:`\phi_{n-1}`
# are the basis functions of the finite element space. A smaller value of
# :math:`\Lambda` indicates a better set of basis functions.
#
# In this demo, we look at how the Lebesgue constant can be approximated using
# Basix, and how variants of Lagrange elements that have lower Lebesgue constants
# can be created.
#
# We begin by importing Basix and Numpy.

import basix
import numpy as np
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType

# In this demo, we consider Lagrange elements defined on a triangle. We start
# by creating a degree 15 Lagrange element that uses equally spaced points.
# This element will exhibit Runge's phenomenon, so we expect a large Lebesgue
# constant.

lagrange = basix.create_element(
    ElementFamily.P, CellType.triangle, 15, LagrangeVariant.equispaced)

# To estimate the Lebesgue constant, we create a lattice of points on the
# triangle and compute
#
# .. math::
#    \Lambda \approx \max_{x\in L}\left(\sum_i\left|\phi_i(x)\right|\right),
#
# where :math:`L` is the set of points in our lattice. As :math:`L` is a
# subset of :math:`R`, the values we compute will be lower bounds of the true
# Lebesgue constants.
#
# The function `create_lattice` takes four inputs: the cell type, the number
# of points in each direction, the lattice type (in this example, we use an
# equally spaced lattice), and a bool indicating whether or not points on the
# boundary should be included. We tabulate our element at the points in the
# lattice then use Numpy to compute the max of the sum.
#
# As expected, the value is large.

points = basix.create_lattice(
    CellType.triangle, 50, LatticeType.equispaced, True)
tab = lagrange.tabulate(0, points)[0]
print(max(np.sum(np.abs(tab), axis=0)))

# A Lagrange element with a lower Lebesgue constant can be created by placing
# the DOFs at Gauss-Lobatto-Legendre (GLL) points. Passing
# `LagrangeVariant.gll_warped` into `create_element` will make an element that
# places its DOF points at warped GLL points on the triangle, as described in
# `Nodal Discontinuous Galerkin Methods (Hesthaven, Warburton, 2008, pp
# 175-180) <https://doi.org/10.1007/978-0-387-72067-8>`_.
#
# The Lebesgue constant for this variant of the element is much smaller than
# for the equally spaced element.

gll = basix.create_element(
    ElementFamily.P, CellType.triangle, 15, LagrangeVariant.gll_warped)
print(max(np.sum(np.abs(gll.tabulate(0, points)[0]), axis=0)))

# An even lower Lebesgue constant can be obtained by placing the DOF points
# at GLL points mapped onto a triangle following the method proposed in
# `Recursive, Parameter-Free, Explicitly Defined Interpolation Nodes for
# Simplices (Isaac, 2020) <https://doi.org/10.1137/20M1321802>`_.

gll2 = basix.create_element(
    ElementFamily.P, CellType.triangle, 15, LagrangeVariant.gll_isaac)
print(max(np.sum(np.abs(gll2.tabulate(0, points)[0]), axis=0)))
