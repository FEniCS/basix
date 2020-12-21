# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

import basix
import numpy as np


def test_gll_warped_pyramid():
    n = 8

    # Check that all the surface points of the pyramid match up with the same points on
    # quad and triangle
    tri_pts = basix.create_lattice(basix.CellType.triangle, n, basix.LatticeType.gll_warped, True)
    quad_pts = basix.create_lattice(basix.CellType.quadrilateral, n, basix.LatticeType.gll_warped, True)
    pyr_pts = basix.create_lattice(basix.CellType.pyramid, n, basix.LatticeType.gll_warped, True)

    # Remove any near-zero values to make sorting robust
    pyr_pts[np.where(abs(pyr_pts) < 1e-12)] = 0.0
    tri_pts[np.where(abs(tri_pts) < 1e-12)] = 0.0
    quad_pts[np.where(abs(quad_pts) < 1e-12)] = 0.0

    idx = np.where(np.isclose(pyr_pts[:, 0], 0.0))
    pyr_x0 = pyr_pts[idx][:, 1:]
    assert np.allclose(np.sort(tri_pts), np.sort(pyr_x0))

    idx = np.where(np.isclose(pyr_pts[:, 0] + pyr_pts[:, 2], 1.0))
    pyr_xz = pyr_pts[idx][:, 1:]
    assert np.allclose(np.sort(tri_pts), np.sort(pyr_xz))

    idx = np.where(np.isclose(pyr_pts[:, 1], 0.0))
    pyr_y0 = pyr_pts[idx][:, 0::2]
    assert np.allclose(np.sort(tri_pts), np.sort(pyr_y0))

    idx = np.where(np.isclose(pyr_pts[:, 1] + pyr_pts[:, 2], 1.0))
    pyr_yz = pyr_pts[idx][:, 0::2]
    assert np.allclose(np.sort(tri_pts), np.sort(pyr_yz))

    idx = np.where(np.isclose(pyr_pts[:, 2], 0.0))
    pyr_z0 = pyr_pts[idx][:, :2]
    assert np.allclose(np.sort(quad_pts), np.sort(pyr_z0))


def test_gll_warped_tetrahedron():
    n = 8

    # Check that all the surface points of the tet match up with the same points on
    # triangle
    tri_pts = basix.create_lattice(basix.CellType.triangle, n, basix.LatticeType.gll_warped, True)
    tet_pts = basix.create_lattice(basix.CellType.tetrahedron, n, basix.LatticeType.gll_warped, True)

    tet_pts[np.where(abs(tet_pts) < 1e-12)] = 0.0
    tri_pts[np.where(abs(tri_pts) < 1e-12)] = 0.0

    idx = np.where(np.isclose(tet_pts[:, 0], 0.0))
    tet_x0 = tet_pts[idx][:, 1:]
    assert np.allclose(np.sort(tri_pts), np.sort(tet_x0))

    idx = np.where(np.isclose(tet_pts[:, 1], 0.0))
    tet_y0 = tet_pts[idx][:, 0::2]
    assert np.allclose(np.sort(tri_pts), np.sort(tet_y0))

    idx = np.where(np.isclose(tet_pts[:, 2], 0.0))
    tet_z0 = tet_pts[idx][:, :2]
    assert np.allclose(np.sort(tri_pts), np.sort(tet_z0))

    # Project x+y+z=1 onto x=0
    idx = np.where(np.isclose(tet_pts[:, 0] + tet_pts[:, 1] + tet_pts[:, 2], 1.0))
    tet_xyz = tet_pts[idx][:, 1:]
    assert np.allclose(np.sort(tri_pts), np.sort(tet_xyz))
