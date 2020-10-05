import numpy
import matplotlib.pyplot as plt
from fiatx import Lagrange, CellType, create_lattice

d = 3
L = Lagrange(CellType.triangle, d)

pts = create_lattice(CellType.triangle, 50, True)

for kk in range(3):
    w = L.tabulate_basis_derivatives(1, pts)[kk]


    ww = 1
    nn = (d + 2) * (d + 1) // 2
    if (nn % 4 == 0 and nn > 4):
        nn //= 4
        ww = 4
    elif (nn % 3 == 0 and nn > 3):
        nn //= 3
        ww = 3
    elif (nn % 2 == 0 and nn > 2):
        nn //= 2
        ww = 2

    plt.figure()
    fig, ax = plt.subplots(ww, nn)
    for j, a in enumerate(ax.flatten()):
        shape_fn = w[:, j]
        print(numpy.max(shape_fn), numpy.min(shape_fn))
        a.tricontourf(pts[:, 0], pts[:, 1], shape_fn)


plt.show()
