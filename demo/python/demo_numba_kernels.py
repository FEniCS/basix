# ==================================
# Calling Basix from Numba kernels
# ==================================
#
# Tabulating an element one point at a time from Python is dominated by
# the cost of the Python call, not by the mathematics. A loop that
# evaluates basis functions at a large number of scattered points can
# therefore be far slower than the finite element solve that produced
# the data.
#
# Basix exposes its tabulation and its push-forward/pull-back maps as
# native functions that can be called directly from a Numba kernel
# compiled in ``nopython`` mode, so that such a loop runs at native
# speed. In this demo we use them to assemble the matrix that
# interpolates a finite element function at a set of arbitrary points.
#
# The helpers live in :mod:`basix.numba_helpers` and require Numba.

import time

import numba
import numpy as np

import basix
from basix import CellType, ElementFamily, LagrangeVariant
from basix.numba_helpers import ElementHandle, map_fn, tabulate_fn

# A mesh of the unit square
# =========================
#
# Basix has no mesh data structure of its own, so we build a small
# triangular mesh by hand: a grid of squares, each split into two
# triangles. ``points`` holds the vertex coordinates and ``cells`` holds
# the three vertices of each triangle.

n = 16
grid = np.linspace(0.0, 1.0, n + 1)
X, Y = np.meshgrid(grid, grid, indexing="ij")
points = np.column_stack([X.ravel(), Y.ravel()])


def vertex(i, j):
    """Index of the mesh vertex at grid position (i, j)."""
    return i * (n + 1) + j


cells = np.array(
    [
        c
        for i in range(n)
        for j in range(n)
        for c in (
            [vertex(i, j), vertex(i + 1, j), vertex(i, j + 1)],
            [vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)],
        )
    ],
    dtype=np.int32,
)

# A degree 2 Lagrange space
# =========================
#
# We use a degree 2 Lagrange element. Its DOFs are point evaluations at
# the three vertices and the three edge midpoints of a triangle, so a
# DOF map for the mesh associates one DOF with each mesh vertex and one
# with each mesh edge. We number the vertex DOFs first, then the edge
# DOFs.

element = basix.create_element(ElementFamily.P, CellType.triangle, 2, LagrangeVariant.gll_isaac)

# The local edges of a triangle, as pairs of local vertices
local_edges = basix.topology(CellType.triangle)[1]

edge_number = {}
dofmap = np.zeros((cells.shape[0], element.dim), dtype=np.int32)
for c, cell in enumerate(cells):
    dofmap[c, :3] = cell
    for local_edge, (a, b) in enumerate(local_edges):
        edge = (min(cell[a], cell[b]), max(cell[a], cell[b]))
        if edge not in edge_number:
            edge_number[edge] = len(points) + len(edge_number)
        dofmap[c, 3 + local_edge] = edge_number[edge]

num_dofs = len(points) + len(edge_number)

# Points to interpolate at
# ========================
#
# We take a large number of points scattered through the mesh. A real
# application would locate each point in the mesh with a bounding box
# tree; here we pick a cell and a position within it, so that the cell
# containing each point is known.

num_points = 200000
rng = np.random.default_rng(42)
point_cells = rng.integers(0, cells.shape[0], num_points).astype(np.int32)

# Random barycentric coordinates, from a uniform point in the unit
# square folded onto the reference triangle
s, t = rng.random((2, num_points))
fold = s + t > 1.0
s[fold], t[fold] = 1.0 - s[fold], 1.0 - t[fold]

v0 = points[cells[point_cells, 0]]
v1 = points[cells[point_cells, 1]]
v2 = points[cells[point_cells, 2]]
x = v0 + s[:, None] * (v1 - v0) + t[:, None] * (v2 - v0)

# The interpolation kernel
# ========================
#
# :func:`basix.numba_helpers.tabulate_fn` returns a tabulation function
# that a Numba kernel can call. It takes the address of an element, the
# number of derivatives, the address of the reference points, the number
# of points, and the address of the output buffer. Addresses of NumPy
# arrays are obtained inside a kernel with ``array.ctypes.data``, and
# the function returns a non-zero value if the call failed.
#
# The function must be referenced as a global (or a closure variable) by
# the kernel, so we bind it before defining the kernel.

tabulate = tabulate_fn(element.dtype)

# The kernel loops over the points. For each one it pulls the point back
# to the reference triangle, evaluates the basis functions there, and
# scatters them into the row of the matrix belonging to that point. The
# cells are affine, so the pull-back is the inverse of the Jacobian
# ``J = [v1 - v0, v2 - v0]`` applied to ``x - v0``.
#
# Because the tabulation function does not touch the Python interpreter,
# the kernel can release the GIL.


@numba.njit(cache=True, nogil=True)
def interpolation_matrix(address, x, point_cells, points, cells, dofmap, A):
    """Assemble the rows of a dense interpolation matrix."""
    X = np.zeros((1, 2))
    basis = np.zeros(dofmap.shape[1])
    for k in range(x.shape[0]):
        c = point_cells[k]
        v0 = points[cells[c, 0]]
        j00 = points[cells[c, 1], 0] - v0[0]
        j10 = points[cells[c, 1], 1] - v0[1]
        j01 = points[cells[c, 2], 0] - v0[0]
        j11 = points[cells[c, 2], 1] - v0[1]
        detJ = j00 * j11 - j01 * j10

        # X = J^-1 (x - v0)
        dx0 = x[k, 0] - v0[0]
        dx1 = x[k, 1] - v0[1]
        X[0, 0] = (j11 * dx0 - j01 * dx1) / detJ
        X[0, 1] = (-j10 * dx0 + j00 * dx1) / detJ

        if tabulate(address, 0, X.ctypes.data, 1, basis.ctypes.data) != 0:
            raise RuntimeError("Basix tabulation failed.")

        for i in range(dofmap.shape[1]):
            A[k, dofmap[c, i]] = basis[i]


# The element is passed to the kernel as the address of the underlying
# C++ object. :class:`basix.numba_helpers.ElementHandle` holds that
# address together with a reference to the element, so that the element
# cannot be garbage collected while a kernel is still using it. The
# handle must be kept alive for as long as the kernel may run.

handle = ElementHandle(element)

A = np.zeros((num_points, num_dofs))
interpolation_matrix(handle.address, x, point_cells, points, cells, dofmap, A)

# Checking the result
# ===================
#
# A degree 2 space reproduces any quadratic exactly, so interpolating
# one must return its values at the points. We take the DOF values by
# evaluating the function at the physical position of each DOF point,
# which we get by mapping the reference DOF points of the element
# through the affine map of each cell.


def u(x):
    """A quadratic function."""
    return 1.0 + 2.0 * x[:, 0] + 3.0 * x[:, 1] + 4.0 * x[:, 0] ** 2 + 5.0 * x[:, 1] ** 2


dof_values = np.zeros(num_dofs)
for c, cell in enumerate(cells):
    v0, v1, v2 = points[cell]
    dof_x = v0 + np.outer(element.points[:, 0], v1 - v0) + np.outer(element.points[:, 1], v2 - v0)
    dof_values[dofmap[c]] = u(dof_x)

assert np.allclose(A @ dof_values, u(x))
print(f"Interpolated {num_points} points exactly.")

# Comparing with a Python loop
# ============================
#
# The same assembly written as a Python loop calls `element.tabulate`
# once per point. We time a smaller number of points and scale, since
# the Python version is considerably slower.

start = time.perf_counter()
interpolation_matrix(handle.address, x, point_cells, points, cells, dofmap, A)
numba_time = (time.perf_counter() - start) / num_points

num_python_points = 20000
start = time.perf_counter()
for k in range(num_python_points):
    c = point_cells[k]
    v0, v1, v2 = points[cells[c]]
    J = np.column_stack([v1 - v0, v2 - v0])
    X = np.linalg.solve(J, x[k] - v0).reshape(1, 2)
    A[k, dofmap[c]] = element.tabulate(0, X)[0, 0, :, 0]
python_time = (time.perf_counter() - start) / num_python_points

print(f"Numba kernel: {numba_time * 1e6:.2f} us/point")
print(f"Python loop:  {python_time * 1e6:.2f} us/point")
print(f"Speedup:      {python_time / numba_time:.1f}x")

# Mapping values to a physical cell
# =================================
#
# Elements whose basis functions are not mapped by the identity need
# their reference values pushing forward to the physical cell.
# :func:`basix.numba_helpers.map_fn` returns a function that applies the
# map of an element, for groups of points that share a Jacobian.
#
# Its arguments are the element address, a flag selecting the direction
# (0 to push forward, 1 to pull back), the number of Jacobians, the
# number of points per Jacobian, the geometric dimension, and then the
# addresses of the values, the Jacobians, their determinants, their
# inverses, and the output buffer.
#
# Unlike `FiniteElement.push_forward`, this function writes into a
# buffer supplied by the caller and allocates no memory, so it can be
# called in a loop without the allocator becoming the bottleneck.

rt = basix.create_element(ElementFamily.RT, CellType.triangle, 1)
rt_handle = ElementHandle(rt)
apply_map = map_fn(rt.dtype)


@numba.njit(cache=True, nogil=True)
def push_forward(address, num_cells, num_points, gdim, U, J, detJ, K, u):
    """Push reference values forward to each physical cell."""
    return apply_map(
        address,
        0,
        num_cells,
        num_points,
        gdim,
        U.ctypes.data,
        J.ctypes.data,
        detJ.ctypes.data,
        K.ctypes.data,
        u.ctypes.data,
    )


# Tabulate the Raviart-Thomas basis at the midpoint of the reference
# triangle, then push it forward to the first few cells of the mesh.

num_mapped_cells = 8
X = np.array([[1.0 / 3.0, 1.0 / 3.0]])
reference = rt.tabulate(0, X)[0, 0]

J = np.zeros((num_mapped_cells, 2, 2))
for c in range(num_mapped_cells):
    v0, v1, v2 = points[cells[c]]
    J[c] = np.column_stack([v1 - v0, v2 - v0])
detJ = np.array([np.linalg.det(J[c]) for c in range(num_mapped_cells)])
K = np.array([np.linalg.inv(J[c]) for c in range(num_mapped_cells)])

# One group of values per cell, with `rt.dim` values in each group
U = np.ascontiguousarray(np.broadcast_to(reference, (num_mapped_cells, rt.dim, 2)))
mapped = np.zeros_like(U)
assert push_forward(rt_handle.address, num_mapped_cells, rt.dim, 2, U, J, detJ, K, mapped) == 0

assert np.allclose(mapped, rt.push_forward(U, J, detJ, K))
print(f"Pushed the Raviart-Thomas basis forward to {num_mapped_cells} cells.")
