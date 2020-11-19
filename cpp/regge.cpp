// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "regge.h"
#include "lattice.h"
#include "polynomial-set.h"
#include <iostream>

using namespace libtab;

namespace
{
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_regge_space(cell::Type celltype, int degree)
{

  if (celltype != cell::Type::triangle and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Unsupported celltype");

  const int tdim = cell::topological_dimension(celltype);
  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size = polyset::size(celltype, degree);
  const int ndofs = basis_size * nc;
  const int psize = basis_size * tdim * tdim;

  Eigen::ArrayXXd wcoeffs = Eigen::ArrayXXd::Zero(ndofs, psize);
  int s = basis_size;
  for (int i = 0; i < tdim; ++i)
  {
    for (int j = 0; j < tdim; ++j)
    {
      int xoff = i + tdim * j;
      int yoff = i + j;
      if (tdim == 3 and i > 0 and j > 0)
        ++yoff;

      wcoeffs.block(yoff * s, xoff * s, s, s) = Eigen::MatrixXd::Identity(s, s);
    }
  }

  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_regge_dual(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);

  const int basis_size = polyset::size(celltype, degree);

  const int ndofs = basis_size * (tdim + 1) * tdim / 2;
  const int space_size = basis_size * tdim * tdim;

  Eigen::ArrayXXd dualmat(ndofs, space_size);
  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const Eigen::ArrayXXd geometry = cell::geometry(celltype);

  // dof counter
  int dof = 0;
  for (std::size_t dim = 1; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const Eigen::ArrayXXd entity_geom
          = cell::sub_entity_geometry(celltype, dim, i);

      Eigen::ArrayXd point = entity_geom.row(0);
      cell::Type ct = cell::sub_entity_type(celltype, dim, i);
      Eigen::ArrayXXd lattice
          = lattice::create(ct, degree + 2, lattice::Type::equispaced, false);
      Eigen::ArrayXXd pts(lattice.rows(), entity_geom.cols());
      for (int j = 0; j < lattice.rows(); ++j)
      {
        pts.row(j) = entity_geom.row(0);
        for (int k = 0; k < entity_geom.rows() - 1; ++k)
        {
          pts.row(j)
              += (entity_geom.row(k + 1) - entity_geom.row(0)) * lattice(j, k);
        }
      }

      Eigen::MatrixXd basis = polyset::tabulate(celltype, degree, 0, pts)[0];

      // Store up outer(t, t) for all tangents
      std::vector<int>& vert_ids = topology[dim][i];
      int ntangents = dim * (dim + 1) / 2;
      std::vector<Eigen::MatrixXd> vvt(ntangents);
      int c = 0;
      for (std::size_t s = 0; s < dim; ++s)
      {
        for (std::size_t d = s + 1; d < dim + 1; ++d)
        {
          const Eigen::VectorXd edge_t
              = geometry.row(vert_ids[d]) - geometry.row(vert_ids[s]);
          // outer product v.v^T
          vvt[c++] = edge_t * edge_t.transpose();
        }
      }

      for (int k = 0; k < pts.rows(); ++k)
      {
        for (int j = 0; j < ntangents; ++j)
        {
          Eigen::Map<Eigen::VectorXd> vvt_flat(vvt[j].data(),
                                               vvt[j].rows() * vvt[j].cols());
          // outer product: outer(outer(t, t), basis)
          const Eigen::MatrixXd vvt_b = vvt_flat * basis.row(k);

          // Copy tensor values row by row into dualmat
          for (int r = 0; r < vvt_b.rows(); ++r)
            dualmat.block(dof, r * vvt_b.cols(), 1, vvt_b.cols())
                = vvt_b.row(r);
          ++dof;
        }
      }
    }
  }

  return dualmat;
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
FiniteElement Regge::create(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);

  Eigen::MatrixXd wcoeffs = create_regge_space(celltype, degree);
  Eigen::MatrixXd dual = create_regge_dual(celltype, degree);

  // TODO
  const int ndofs = dual.rows();
  int perm_count = 0;
  std::vector<Eigen::MatrixXd> base_permutations(
      perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::MatrixXd coeffs
      = FiniteElement::compute_expansion_coefficients(wcoeffs, dual);

  // Regge has (d+1) dofs on each edge, 3d(d+1)/2 on each face
  // and d(d-1)(d+1) on the interior in 3D
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), degree + 1);
  entity_dofs[2].resize(topology[2].size(), 3 * (degree + 1) * degree / 2);
  if (tdim > 2)
    entity_dofs[3] = {(degree + 1) * degree * (degree - 1)};

  return FiniteElement(Regge::family_name, celltype, degree, {tdim, tdim},
                       coeffs, entity_dofs, base_permutations);
}
//-----------------------------------------------------------------------------
