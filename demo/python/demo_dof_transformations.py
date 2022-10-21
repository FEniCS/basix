# ====================================
# DOF permutations and transformations
# ====================================
#
# When using high degree finite elements on general meshes, adjustments
# may need to be made to correct for differences in the orientation of
# mesh entities on the mesh and on the reference cell. For example, in
# a degree 4 Lagrange element on a triangle, there are 3 degrees of
# freedom (DOFs) associated with each edge. If two neighbouring cells in
# a mesh disagree on the direction of the edge, they could put an
# incorrectly combine the local basis functions to give the wrong global
# basis function.
#
# This issue and the use of permutations and transformations to correct
# it is discussed in detail in `Construction of arbitrary order finite
# element degree-of-freedom maps on polygonal and polyhedral cell
# meshes (Scroggs, Dokken, Richardons, Wells,
# 2021) <https://arxiv.org/abs/2102.11901>`_.
#
# Functions to permute and transform high degree elements are
# provided by Basix. In this demo, we show how these can be used.
#
# First, we import Basix and Numpy.

import basix
import numpy as np
from basix import ElementFamily, CellType, LagrangeVariant, LatticeType

# Degree 5 Lagrange element
# =========================
#
# We create a degree 5 Lagrange element on a triangle, then print the
# values of the attributes `dof_transformations_are_identity` and
# `dof_transformations_are_permutations`.
#
# The value of `dof_transformations_are_identity` is False: this tells
# us that permutations or transformations are needed for this element.
#
# The value of `dof_transformations_are_permutations` is True: this
# tells us that for this element, all the corrections we need to apply
# permutations. This is the simpler case, and means we make the
# orientation corrections by applying permutations when creating the
# DOF map.

lagrange = basix.create_element(
    ElementFamily.P, CellType.triangle, 5, LagrangeVariant.equispaced)
print(lagrange.dof_transformations_are_identity)
print(lagrange.dof_transformations_are_permutations)

# We can apply permutations by using the matrices returned by the
# method `base_transformations`. This method will return one matrix
# for each edge of the cell (for 2D and 3D cells), and two matrices
# for each face of the cell (for 3D cells). These describe the effect
# of reversing the edge or reflecting and rotating the face.
#
# For this element, we know that the base transformations will be
# permutation matrices.

print(lagrange.base_transformations())

# The matrices returned by `base_transformations` are quite large, and
# are equal to the identity matrix except for a small block of the
# matrix. It is often easier and more efficient to use the matrices
# returned by the method `entity_transformations` instead.
#
# `entity_transformations` returns a dictionary that maps the type
# of entity (`"interval"`, `"triangle"`, `"quadrilateral"`) to a
# matrix describing the effect of permuting that entity on the DOFs
# on that entity.
#
# For this element, we see that this method returns one matrix for
# an interval: this matrix reverses the order of the four DOFs
# associated with that edge.

print(lagrange.entity_transformations())

# In orders to work out which DOFs are associated with each edge,
# we use the attribute `entity_dofs`. For example, the following can
# be used to see which DOF numbers are associated with edge (dim 1)
# number 2:

print(lagrange.entity_dofs[1][2])

# Degree 2 Lagrange element
# =========================
#
# For a degree 2 Lagrange element, no permutations or transformations
# are needed. We can verify this by checking that
# `dof_transformations_are_identity` is `True`. To confirm that the
# transformations are identity matrices, we also print the base
# transformations.

lagrange_degree_2 = basix.create_element(
    ElementFamily.P, CellType.triangle, 2, LagrangeVariant.equispaced)
print(lagrange_degree_2.dof_transformations_are_identity)
print(lagrange_degree_2.base_transformations())

# Degree 2 Nédélec element
# ========================
#
# For a degree 2 Nédélec (first kind) element on a tetrahedron, the
# corrections are not all permutations, so both
# `dof_transformations_are_identity` and
# `dof_transformations_are_permutations` are `False`.

nedelec = basix.create_element(ElementFamily.N1E, CellType.tetrahedron, 2)
print(nedelec.dof_transformations_are_identity)
print(nedelec.dof_transformations_are_permutations)

# For this element, `entity_transformations` returns a dictionary
# with two entries: a matrix for an interval that describes
# the effect of reversing the edge; and an array of two matrices
# for a triangle. The first matrix for the triangle describes
# the effect of rotating the triangle. The second matrix describes
# the effect of reflecting the triangle.
#
# For this element, the matrix describing the effect of rotating
# the triangle is
#
# .. math::
#    \left(\begin{array}{cc}-1&-1\\1&0\end{array}\right).
#
# This is not a permutation, so this must be applied when assembling
# a form and cannot be applied to the DOF numbering in the DOF map.

print(nedelec.entity_transformations())

# To demonstrate how these transformations can be used, we create a
# lattice of points where we will tabulate the element.

points = basix.create_lattice(
    CellType.tetrahedron, 5, LatticeType.equispaced, True)

# If (for example) the direction of edge 2 in the physical cell does
# not match its direction on the reference, then we need to adjust the
# tabulated data.
#
# As the cell sub-entity that we are correcting is an interval, we
# get the `"interval"` item from the entity transformations dictionary.
# We use `entity_dofs[1][2]` (1 is the dimension of an edge, 2 is the
# index of the edge we are reversing) to find out which dofs are on
# our edge.
#
# To adjust the tabulated data, we loop over each point in the lattice
# and over the value size. For each of these values, we apply the
# transformation matrix to the relevant DOFs.

data = nedelec.tabulate(0, points)

transformation = nedelec.entity_transformations()["interval"][0]
dofs = nedelec.entity_dofs[1][2]

for point in range(data.shape[1]):
    for dim in range(data.shape[3]):
        data[0, point, dofs, dim] = np.dot(transformation, data[0, point, dofs, dim])

print(data)

# More efficient functions that apply the transformations and
# permutations directly to data can be used via Basix's C++
# interface.
#
# C++ demo
# ========
# The following C++ code runs the same demo using Basix's C++ interface:
#
# .. literalinclude:: ../cpp/demo_dof_transformations/main.cpp
#    :language: c++
