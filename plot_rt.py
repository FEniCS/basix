import matplotlib.pyplot as plt
from libtab import RaviartThomas, create_lattice, CellType


RT = RaviartThomas("triangle", 3)
pts = create_lattice(CellType.triangle, 20, True)

w = RT.tabulate(0, pts)[0]
nc = w.shape[1]//2
print(w.shape)

fig, ax = plt.subplots(3, 5)
for j, a in enumerate(ax.flatten()):
    ned_shape_fn_x = w[:, j]
    ned_shape_fn_y = w[:, j + nc]
    a.quiver(pts[:, 0], pts[:, 1], ned_shape_fn_x, ned_shape_fn_y)

plt.show()
