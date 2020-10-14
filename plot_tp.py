import numpy
import matplotlib.pyplot as plt
from libtab import TensorProduct, create_lattice, CellType

d = 4
L = TensorProduct(CellType.quadrilateral, d)
pts = create_lattice(CellType.quadrilateral, 50, True)

w = L.tabulate(0, pts)[0]

ww = 1
nn = (d + 1) * (d + 1)
if (nn % 5 == 0 and nn > 5):
    nn //= 5
    ww = 5
elif (nn % 4 == 0 and nn > 4):
    nn //= 4
    ww = 4
elif (nn % 3 == 0 and nn > 3):
    nn //= 3
    ww = 3
elif (nn % 2 == 0 and nn > 2):
    nn //= 2
    ww = 2

fig, ax = plt.subplots(ww, nn)
for j, a in enumerate(ax.flatten()):
    shape_fn = w[:, j]
    print(numpy.max(shape_fn), numpy.min(shape_fn))
    a.tricontourf(pts[:, 0], pts[:, 1], shape_fn,
                  levels=numpy.arange(-.6, 1.2, 0.1))

plt.show()
