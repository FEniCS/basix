import pytest
import ufl
import basix
import basix.ufl_wrapper
import numpy as np


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_element(inputs):
    basix.ufl_wrapper.create_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_vector_element(inputs):
    basix.ufl_wrapper.create_vector_element(*inputs)


@pytest.mark.parametrize("inputs", [
    ("Lagrange", "triangle", 2),
    ("Lagrange", basix.CellType.triangle, 2),
    (basix.ElementFamily.P, basix.CellType.triangle, 2),
    (basix.ElementFamily.P, "triangle", 2),
])
def test_create_tensor_element(inputs):
    basix.ufl_wrapper.create_tensor_element(*inputs)


@pytest.mark.parametrize("e", [
    ufl.FiniteElement("Q", "quadrilateral", 1),
    ufl.FiniteElement("Lagrange", "triangle", 2),
    ufl.VectorElement("Lagrange", "triangle", 2),
    ufl.TensorElement("Lagrange", "triangle", 2),
    ufl.MixedElement(ufl.VectorElement("Lagrange", "triangle", 2), ufl.VectorElement("Lagrange", "triangle", 1)),
    ufl.EnrichedElement(ufl.FiniteElement("Lagrange", "triangle", 1), ufl.FiniteElement("Bubble", "triangle", 3)),
    ufl.EnrichedElement(ufl.VectorElement("Lagrange", "triangle", 1), ufl.VectorElement("Bubble", "triangle", 3)),
])
def test_convert_ufl_element(e):
    e2 = basix.ufl_wrapper.convert_ufl_element(e)
    # Check that element is hashable
    hash(e2)


@pytest.mark.parametrize("celltype, family, degree, variants", [
    ("Lagrange", "triangle", 1, []),
    ("Lagrange", "triangle", 3, [basix.LagrangeVariant.gll_warped]),
    ("Lagrange", "tetrahedron", 2, [])
])
def test_converted_elements(celltype, family, degree, variants):
    e1 = basix.ufl_wrapper.create_element(celltype, family, degree, *variants)
    e2 = ufl.FiniteElement(celltype, family, degree)

    assert e1 == basix.ufl_wrapper.convert_ufl_element(e1)
    assert e1 == basix.ufl_wrapper.convert_ufl_element(e2)

    e1 = basix.ufl_wrapper.create_vector_element(celltype, family, degree, *variants)
    e2 = ufl.VectorElement(celltype, family, degree)

    assert e1 == basix.ufl_wrapper.convert_ufl_element(e1)
    assert e1 == basix.ufl_wrapper.convert_ufl_element(e2)


# Custom element: P2 bubbles on each facet of a triangle
wcoeffs = np.zeros((3, 6))
pts, wts = basix.make_quadrature(basix.CellType.triangle, 4)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 2, pts)
x = pts[:, 0]
y = pts[:, 1]
for i in range(6):
    wcoeffs[0, i] = sum(x * y * poly[i, :] * wts)
    wcoeffs[1, i] = sum(x * (1 - x - y) * poly[i, :] * wts)
    wcoeffs[2, i] = sum((1 - x - y) * y * poly[i, :] * wts)

x = [[], [], [], []]
for _ in range(3):
    x[0].append(np.zeros((0, 2)))
x[1].append(np.array([[0.5, 0.5]]))
x[1].append(np.array([[0.0, 0.5]]))
x[1].append(np.array([[0.5, 0.0]]))
x[2].append(np.zeros((0, 2)))

M = [[], [], [], []]
for _ in range(3):
    M[0].append(np.zeros((0, 1, 0, 1)))
for _ in range(3):
    M[1].append(np.array([[[[1.]]]]))
M[2].append(np.zeros((0, 1, 0, 1)))

facet_bubbles = basix.ufl_wrapper.BasixElement(basix.create_custom_element(
    basix.CellType.triangle, [], wcoeffs, x, M, 0, basix.MapType.identity, False, -1, 2))

# Custom element: a single P4 bubble on an interval
wcoeffs = np.zeros((1, 5))
pts, wts = basix.make_quadrature(basix.CellType.interval, 8)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.interval, 4, pts)
x = pts[:, 0]
f = (x * (1 - x)) ** 2
for i in range(5):
    wcoeffs[0, i] = sum(f * poly[i, :] * wts)

x = [[], [], [], []]
for _ in range(2):
    x[0].append(np.zeros((0, 1)))
x[1].append(np.array([[0.5]]))

M = [[], [], [], []]
for _ in range(2):
    M[0].append(np.zeros((0, 1, 0, 1)))
M[1].append(np.array([[[[1.]]]]))

p4_bubble = basix.ufl_wrapper.BasixElement(basix.create_custom_element(
    basix.CellType.interval, [], wcoeffs, x, M, 0, basix.MapType.identity, False, -1, 4))

# Custom element: Hdiv bubble
wcoeffs = np.zeros((2, 20))
pts, wts = basix.make_quadrature(basix.CellType.triangle, 6)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 3, pts)
x = pts[:, 0]
y = pts[:, 1]
f = x * y * (1 - x - y)
for i in range(10):
    wcoeffs[0, i] = sum(f * poly[i, :] * wts)
    wcoeffs[1, 10 + i] = sum(f * poly[i, :] * wts)

x = [[], [], [], []]
for _ in range(3):
    x[0].append(np.zeros((0, 2)))
    x[1].append(np.zeros((0, 2)))
x[2].append(np.array([[1 / 3, 1 / 3]]))

M = [[], [], [], []]
for _ in range(3):
    M[0].append(np.zeros((0, 2, 0, 1)))
    M[1].append(np.zeros((0, 2, 0, 1)))
M[2].append([[[[1.]], [[0.]]], [[[0.]], [[1.]]]])

hdiv_bubble = basix.ufl_wrapper.BasixElement(basix.create_custom_element(
    basix.CellType.triangle, [2], wcoeffs, x, M, 0, basix.MapType.contravariantPiola, False, -1, 3))
hcurl_bubble = basix.ufl_wrapper.BasixElement(basix.create_custom_element(
    basix.CellType.triangle, [2], wcoeffs, x, M, 0, basix.MapType.covariantPiola, False, -1, 3))


@pytest.mark.parametrize("elements", [
    [ufl.FiniteElement("Lagrange", "triangle", i), ufl.FiniteElement("Bubble", "triangle", j)]
    for i in [1, 2] for j in [3, 4]
] + [
    [ufl.FiniteElement("Lagrange", "tetrahedron", i), ufl.FiniteElement("Bubble", "tetrahedron", j)]
    for i in [1, 2, 3] for j in [4, 5]
] + [
    [ufl.FiniteElement("Lagrange", "quadrilateral", 1),
     basix.ufl_wrapper.create_element("Bubble", "quadrilateral", j)] for j in [2, 3, 4]
] + [
    [ufl.FiniteElement("Lagrange", "hexahedron", 1),
     basix.ufl_wrapper.create_element("Bubble", "hexahedron", j)] for j in [2, 3, 4]
] + [
    [ufl.FiniteElement("Lagrange", "triangle", 1), facet_bubbles],
    [ufl.VectorElement("Lagrange", "quadrilateral", 1),
     basix.ufl_wrapper.create_vector_element("Bubble", "quadrilateral", 2)],
    [ufl.TensorElement("Lagrange", "quadrilateral", 1),
     basix.ufl_wrapper.create_tensor_element("Bubble", "quadrilateral", 2)],
    [ufl.FiniteElement("Lagrange", "triangle", 1), facet_bubbles, ufl.FiniteElement("Bubble", "triangle", 3)],
    [ufl.FiniteElement("RT", "triangle", 1), hdiv_bubble],
    [ufl.FiniteElement("RT", "triangle", 2), hdiv_bubble],
    [ufl.FiniteElement("N1curl", "triangle", 1), hcurl_bubble],
    [ufl.FiniteElement("N1curl", "triangle", 2), hcurl_bubble],
    # [basix.ufl_wrapper.create_element("Hermite", "interval", 3), p4_bubble],
])
def test_enriched_element(elements):
    converted_elements = [basix.ufl_wrapper.convert_ufl_element(e) for e in elements]
    e1 = basix.ufl_wrapper.create_enriched_element(converted_elements, preserve_dofs=True)
    # Check that element is hashable
    hash(e1)

    e2 = basix.ufl_wrapper.create_enriched_element(converted_elements, preserve_functions=True)
    # Check that element is hashable
    hash(e2)

    points = basix.create_lattice(e2.cell_type, 5, basix.LatticeType.equispaced, True)
    tab = e2.tabulate(0, points)[0]

    tab2 = np.zeros_like(tab)
    ndofs = tab2.shape[1] // elements[0].value_size()
    row = 0
    for e in converted_elements:
        t = e.tabulate(0, points)[0]
        if e.block_size > 1:
            for b in range(e.block_size):
                tab2[:, b * ndofs + row: b * ndofs + row + e.dim] = t[:, b::e.block_size]
            row += e.dim
        elif e.value_size > 1:
            for b in range(e.value_size):
                tab2[:, b * ndofs + row: b * ndofs + row + e.dim] = t[:, b * e.dim: (b + 1) * e.dim]
            row += e.dim
        else:
            tab2[:, row: row + t.shape[1]] = t
            row += t.shape[1]

    assert np.allclose(tab, tab2)
