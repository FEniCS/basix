import numpy
import matplotlib.pyplot as plt
from fiatx import Lagrange, CellType, create_lattice

d = 3
L = Lagrange(CellType.triangle, d)

pts = create_lattice(CellType.triangle, 50, True)
w0 = L.tabulate_basis_derivatives(2, pts)
numpy.set_printoptions(linewidth=240)
for kk in range(6):
    w = w0[kk]
    print(w)

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


    fig, ax = plt.subplots(ww, nn)
    ax[0,0].set_title("w[%d]" % kk)
    for j, a in enumerate(ax.flatten()):
        shape_fn = w[:, j]
        a.tricontourf(pts[:, 0], pts[:, 1], shape_fn)


plt.show()
