# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import libtab


def test_cr2d():
    cr2 = libtab.CrouzeixRaviart(libtab.CellType.triangle, 1)
    pts = libtab.create_lattice(libtab.CellType.triangle, 2, True)
    w = cr2.tabulate(0, pts)[0]
    print(w.shape)


def test_nedelec3d():
    cr3 = libtab.CrouzeixRaviart(libtab.CellType.tetrahedron, 1)
    pts = libtab.create_lattice(libtab.CellType.tetrahedron, 2, True)
    w = cr3.tabulate(0, pts)[0]
    print(w.shape)
