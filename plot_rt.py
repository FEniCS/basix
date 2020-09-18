import numpy
import matplotlib.pyplot as plt
from fiatx import *

RT = RaviartThomas(CellType.triangle, 2)

cell = Cell(CellType.triangle)
pts = cell.create_lattice(20, True)

w = RT.tabulate_basis(pts)
nc = w.shape[1]//2

fig, ax = plt.subplots(2, 3)
for j, a in enumerate(ax.flatten()):
    ned_shape_fn_x = w[:, j]
    ned_shape_fn_y = w[:, j + nc]
    a.quiver(pts[:,0], pts[:, 1], ned_shape_fn_x, ned_shape_fn_y)

plt.show()
