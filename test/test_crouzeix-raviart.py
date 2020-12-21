# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix


def test_cr2d():
    cr2 = basix.CrouzeixRaviart("triangle", 1)
    pts = basix.create_lattice(basix.CellType.triangle, 2, basix.LatticeType.equispaced, True)
    w = cr2.tabulate(0, pts)[0]
    print(w.shape)


def test_cr3d():
    cr3 = basix.CrouzeixRaviart("tetrahedron", 1)
    pts = basix.create_lattice(basix.CellType.tetrahedron, 2, basix.LatticeType.equispaced, True)
    w = cr3.tabulate(0, pts)[0]
    print(w.shape)
