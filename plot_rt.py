import matplotlib.pyplot as plt
from basix import RaviartThomas, create_lattice, CellType, LatticeType


RT = RaviartThomas("triangle", 3)
pts = create_lattice(CellType.triangle, 20, LatticeType.equispaced, True)

w = RT.tabulate(0, pts)[0]
nc = w.shape[1]//2
print(w.shape)

fig, ax = plt.subplots(3, 5)
for j, a in enumerate(ax.flatten()):
    rt_shape_fn_x = w[:, j]
    rt_shape_fn_y = w[:, j + nc]
    a.quiver(pts[:, 0], pts[:, 1], rt_shape_fn_x, rt_shape_fn_y)

plt.show()
