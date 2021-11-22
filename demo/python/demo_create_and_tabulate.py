# ==================================
# Creating and tabulating an element
# ==================================
#
# This demo shows how Basix can be used to create an element
# and tabulate the values of its basis functions at a set of
# points
#
# First, we import Basix and Numpy.

import basix
import numpy as np
from basix import ElementFamily, CellType, LagrangeVariant

# Next, we create a degree 4 Lagrange element on a quadrilateral using the function
# `create_element`. The first input is the element family: for Lagrange elements,
# we use `ElementFamily.P`. The second input is the cell type. The third
# input is the degree of the element. For Lagrange elements, we must provide a
# fourth input: the Lagrange variant. In this example, we use the equispaced
# variant: this will place the degrees of freedom (DOFs) of the element in an
# equally spaced lattice.

lagrange = basix.create_element(
    ElementFamily.P, CellType.quadrilateral, 4, LagrangeVariant.equispaced)

# We now print the number of DOFs that this element has.

print(lagrange.dim)

# We see that the element has 25 DOFs: for this element, there will be 1 DOFs at
# each vertex, 3 DOFs on each edge, and 9 DOFs on the interior of the quadrilateral.
#
# Next, we create a set of points as a numpy array, and tabulate the basis functions
# of the Lagrange space at these points. The first input of `tabulate` is the number
# of derivative to tabulate: we set this to 0 so will to compute the values of the
# functions (and no derivatives). We pass in the points as the second input.

points = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.3], [0.3, 0.6], [0.4, 1.0]])
tab = lagrange.tabulate(0, points)
print(tab)
print(tab.shape)

# The result of tabulating is a 1 by 5 by 25 by 1 Numpy array. The first dimension
# is 1 as we are only tabulating the function values; it would be higher if we
# had asked for derivatives too. The second dimension (5) is the number of points.
# The third dimension (25) is the number of DOFs. The fourth dimension (1) is the
# value size of the element: this will be greater than 1 for vector-values elements.
#
# C++ demo
# ========
# The following C++ code runs the same demo using Basix's C++ interface:
#
# .. literalinclude:: ../cpp/demo_create_and_tabulate/main.cpp
#    :language: c++
