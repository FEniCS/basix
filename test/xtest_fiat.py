import FIAT
import libtab
import pytest
import numpy as np

"""
Some tests against FIAT. Some may fail due to issues in FIAT
(e.g. normalisation of orthonormal polynomial sets when n=0)
"""


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_triangle(order):
    nderivs = 2
    cell = FIAT.ufc_simplex(2)
    L = FIAT.Lagrange(cell, order)

    cell_type = libtab.CellType.triangle
    pts = libtab.create_lattice(cell_type, 3, True)

    tab_fiat = L.tabulate(nderivs, pts)
    L = libtab.Lagrange(cell_type, order)
    tab_libtab = L.tabulate(nderivs, pts)
    print(nderivs, len(tab_libtab))

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    print()
    for p in range(nderivs + 1):
        for q in range(p + 1):
            t = (p - q, q)
            print("idx = ", libtab.index(p - q, q))
            print(tab_fiat[t])
            print(tab_libtab[libtab.index(p - q, q)].transpose())
            print()
            assert(np.isclose(tab_fiat[t],
                              tab_libtab[libtab.index(p - q, q)].transpose()).all())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_triangle_rt(order):
    cell_type = libtab.CellType.triangle
    pts = libtab.create_lattice(cell_type, 2, True)
    nderivs = 2

    cell = FIAT.ufc_simplex(2)
    L = FIAT.RaviartThomas(cell, order, variant='integral')
    tab_fiat = L.tabulate(nderivs, pts)

    L = libtab.RaviartThomas(cell_type, order)
    tab_libtab = L.tabulate(nderivs, pts)
    print(nderivs, len(tab_libtab))

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    print()
    for p in range(nderivs + 1):
        for q in range(p + 1):
            t = (p - q, q)
            print("idx = ", libtab.index(p - q, q))
            tab_fiat_cat = np.vstack((tab_fiat[t][:, 0, :],
                                      tab_fiat[t][:, 1, :]))
#            print(tab_fiat[t][:,0,:])
#            print(tab_fiat[t][:,1,:])
            print(tab_fiat_cat)
            print(tab_libtab[libtab.index(p - q, q)].transpose())
            print()
            print(np.isclose(tab_fiat_cat,
                             tab_libtab[libtab.index(p - q, q)].transpose()))
            assert(np.isclose(tab_fiat_cat,
                              tab_libtab[libtab.index(p - q, q)].transpose()).all())


@pytest.mark.parametrize("order", [1, 2, 3])
def test_triangle_ned(order):
    cell_type = libtab.CellType.triangle
    pts = libtab.create_lattice(cell_type, 2, True)
    nderivs = 2

    cell = FIAT.ufc_simplex(2)
    L = FIAT.Nedelec(cell, order, variant='integral')
    tab_fiat = L.tabulate(nderivs, pts)

    L = libtab.Nedelec(cell_type, order)
    tab_libtab = L.tabulate(nderivs, pts)
    print(nderivs, len(tab_libtab))

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    print()
    for p in range(nderivs + 1):
        for q in range(p + 1):
            t = (p-q, q)
            print("libtab.index = ", libtab.index(p-q, q))
            tab_fiat_cat = np.vstack((tab_fiat[t][:, 0, :],
                                      tab_fiat[t][:, 1, :]))
#            print(tab_fiat[t][:,0,:])
#            print(tab_fiat[t][:,1,:])
            print(tab_fiat_cat)
            print(tab_libtab[libtab.index(p - q, q)].transpose())
            print()
            print(np.isclose(tab_fiat_cat,
                             tab_libtab[libtab.index(p - q, q)].transpose()))
            assert(np.isclose(tab_fiat_cat,
                              tab_libtab[libtab.index(p - q, q)].transpose()).all())


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_tet(order):
    nderivs = order
    cell = FIAT.ufc_simplex(3)
    L = FIAT.Lagrange(cell, order)

    cell_type = libtab.simplex_type(3)
    pts = libtab.create_lattice(cell_type, 7, True)
    print(pts)

    tab_fiat = L.tabulate(nderivs, pts)
    L = libtab.Lagrange(cell_type, order)
    tab_libtab = L.tabulate(nderivs, pts)
    print(nderivs, len(tab_libtab))

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    print()
    for p in range(nderivs + 1):
        for q in range(p + 1):
            for r in range(q + 1):
                t = (p - q, q - r, r)
                print("libtab.index = ", libtab.index(*t))
                print(tab_fiat[t])
                print(tab_libtab[libtab.index(*t)].transpose())
                print()
                assert(np.isclose(tab_fiat[t],
                                  tab_libtab[libtab.index(*t)].transpose()).all())
