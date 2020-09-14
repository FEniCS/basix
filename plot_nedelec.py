import numpy
import matplotlib.pyplot as plt
from fiatx import *

N = Nedelec2D(1)

pts = create_lattice([[0,0],[0,1],[1,0]], 20, True)
w = N.tabulate_basis(pts)

fig, ax = plt.subplots(1, 3)
for j, a in enumerate(ax.flatten()):
    ned_shape_fn_x = w[:, j]
    ned_shape_fn_y = w[:, j + 3]
    a.quiver(pts[:,0], pts[:, 1], ned_shape_fn_x, ned_shape_fn_y)

plt.show()
