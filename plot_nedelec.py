import numpy
import matplotlib.pyplot as plt
from fiatx import *

N = Nedelec2D(2)

pts = make_lattice(20, [[-1,-1],[-1,1],[1,-1]], True)

w = N.tabulate_basis(pts)

fig, ax = plt.subplots(2, 3)
for j, a in enumerate(ax.flatten()):
    ned_shape_fn_x = w[:, j]
    ned_shape_fn_y = w[:, j + 6]
    a.quiver(pts[:,0], pts[:, 1], ned_shape_fn_x, ned_shape_fn_y)

plt.show()
