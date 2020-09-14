import numpy
import matplotlib.pyplot as plt
from fiatx import *

d = 3
L = Lagrange(2, d)

pts = create_lattice([[0,0],[0,1],[1,0]], 50, True)

w = L.tabulate_basis(pts)

ww = 1
nn = (d+2)*(d+1)//2
if (nn % 4 == 0 and nn > 4):
    nn //= 4
    ww = 4
elif (nn % 3 == 0 and nn > 3):
    nn //= 3
    ww = 3
elif (nn %2 == 0 and nn > 2):
    nn //= 2
    ww = 2

fig, ax = plt.subplots(ww, nn)
for j, a in enumerate(ax.flatten()):
    shape_fn = w[:, j]
    print (numpy.max(shape_fn),numpy.min(shape_fn))
    a.tricontourf(pts[:,0], pts[:, 1], shape_fn, levels=numpy.arange(-.6, 1.2, 0.1))

plt.show()
