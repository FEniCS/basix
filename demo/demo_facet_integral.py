# ==========================
# Computing a facet integral
# ==========================
#
# In this demo, we look at how Basix can be used to compute the integral
# of the normal derivative of a basis function over a triangular facet
# of a tetrahedral cell.
#
# As an example, we integrate the normal derivative of the fifth basis
# function (note: counting starts at 0) of a degree 3 Lagrange space
# over the zeroth facet of a tetrahedral cell. This facet will have
# vertices at (1,0,0), (0,1,0) and (0,0,1).
#
# We start by importing Basis and Numpy.

import basix
import numpy as np
from basix import ElementFamily, CellType, LagrangeVariant

# We define a degree 3 Lagrange space on a tetrahedron.

lagrange = basix.create_element(
    ElementFamily.P, CellType.tetrahedron, 3, LagrangeVariant.equispaced)

# The facets of a tetrahedron are triangular, so we create a quadrature
# rule on a triangle. We use an order 3 rule so that we can integrate the
# basis functions in our space exactly.

points, weights = basix.make_quadrature(CellType.triangle, 3)

# Next, we must map the quadrature points to our facet. We use the function
# `geometry` to get the coordinates of the vertices of the tetrahedron, and
# we use `sub_entity_connectivity` to see which vertices are adjacent to
# our facet. We get the sub-entity connectivity using the indices 2 (facets
# of 3D cells have dimension 2), 0 (vertices have dimension 0), and 0 (the
# index of the facet we chose to use).
#
# Using this information, we can map the quadrature points to the facet.

vertices = basix.geometry(CellType.tetrahedron)
facet = basix.cell.sub_entity_connectivity(CellType.tetrahedron)[2][0][0]
mapped_points = np.array([
    vertices[facet[0]] * (1 - x - y) + vertices[facet[1]] * x + vertices[facet[2]] * y
    for x, y in points
])

# We now compute the normal derivative of the fifth basis function at
# the quadrature points. First, we use `facet_outward_normals` to get
# the normal vector to the facet.
#
# We then tabulate the basis functions of our space at the quadrature
# points. We pass 1 in as the first argument, as we want the derivatives
# of the basis functions. The result of tabulation will be an array of
# size 4 by number of quadrature points by number of degrees of freedom.
# To get the data that we want, we use the indices `1:` (to get the
# derivatives and not also the function values), `:` (to include every
# point), 5 (to get the fifth basis function), and 0 (to get the only
# entry as the value size is 1).
#
# We then multiply the three derivatives of the basis function by
# the three componenets of the normal.

normal = basix.cell.facet_outward_normals(CellType.tetrahedron)[0]
tab = lagrange.tabulate(1, mapped_points)[1:, :, 5, 0]
normal_deriv = tab[0] * normal[0] + tab[1] * normal[1] + tab[2] * normal[2]

# As our facet is not the reference triangle, we must multiply the
# integrand by the norm of the Jacobian. We compute this by taking the
# cross product of the two columns of the Jacobian, and then compute
# the integral.

jacobian = basix.cell.facet_jacobians(CellType.tetrahedron)[0]
size_jacobian = np.linalg.norm(np.cross(jacobian[:, 0], jacobian[:, 1]))
print(np.sum(normal_deriv * weights) * size_jacobian)
