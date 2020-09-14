import numpy
import matplotlib.pyplot as plt
from fiatx import *

RT = RaviartThomas(2, 2)

pts = create_lattice([[0,0],[0,1],[1,0]], 20, True)

w = RT.tabulate_basis(pts)
nc = w.shape[1]//2

fig, ax = plt.subplots(2, 3)
for j, a in enumerate(ax.flatten()):
    ned_shape_fn_x = w[:, j]
    ned_shape_fn_y = w[:, j + nc]
    a.quiver(pts[:,0], pts[:, 1], ned_shape_fn_x, ned_shape_fn_y)

plt.show()
