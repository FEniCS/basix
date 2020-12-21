import matplotlib.pyplot as plt
from basix import Nedelec, CellType, LatticeType, create_lattice

N = Nedelec("triangle", 2)
pts = create_lattice(CellType.triangle, 20, LatticeType.equispaced, True)
w = N.tabulate(0, pts)[0]
nc = w.shape[1]//2
fig, ax = plt.subplots(2, 4)
for j, a in enumerate(ax.flatten()):
    ned_shape_fn_x = w[:, j]
    ned_shape_fn_y = w[:, j + nc]
    a.quiver(pts[:, 0], pts[:, 1], ned_shape_fn_x, ned_shape_fn_y)

plt.show()
