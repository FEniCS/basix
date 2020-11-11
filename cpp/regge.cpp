// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "regge.h"
#include "polynomial-set.h"
#include <iostream>

using namespace libtab;

namespace
{

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_regge_space(cell::Type celltype, int degree)
{

  if (celltype != cell::Type::triangle and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Unsupported celltype");

  const int tdim = cell::topological_dimension(celltype);
  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size = polyset::size(celltype, degree);
  const int ndofs = basis_size * nc;
  const int psize = basis_size * tdim * tdim;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs(
      ndofs, psize);
  wcoeffs.setZero();
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
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_regge_dual(cell::Type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);

  const int basis_size = polyset::size(celltype, degree);

  const int ndofs = basis_size * (tdim + 1) * tdim / 2;
  const int space_size = basis_size * tdim * tdim;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat(
      ndofs, space_size);
  auto topology = cell::topology(celltype);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      geometry = cell::geometry(celltype);

  // dof counter
  int dof = 0;
  for (std::size_t dim = 1; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          entity_geom = cell::sub_entity_geometry(celltype, dim, i);

      Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> point
          = entity_geom.row(0);
      cell::Type ct = cell::simplex_type(dim);
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          lattice = cell::create_lattice(ct, degree + 2, false);
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts(
          lattice.rows(), entity_geom.cols());

      for (int j = 0; j < lattice.rows(); ++j)
      {
        pts.row(j) = entity_geom.row(0);
        for (int k = 0; k < entity_geom.rows() - 1; ++k)
        {
          pts.row(j)
              += (entity_geom.row(k + 1) - entity_geom.row(0)) * lattice(j, k);
        }
      }

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          basis = polyset::tabulate(celltype, degree, 0, pts)[0];

      // Store up outer(t, t) for all tangents
      std::vector<int>& vert_ids = topology[dim][i];
      int ntangents = dim * (dim + 1) / 2;
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
          vvt(ntangents);
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
          const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>
              vvt_b = vvt_flat * basis.row(k);
          dualmat.row(dof++) = Eigen::Map<const Eigen::RowVectorXd>(
              vvt_b.data(), vvt_b.rows() * vvt_b.cols());
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

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs
      = create_regge_space(celltype, degree);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat
      = create_regge_dual(celltype, degree);

  // TODO
  const int ndofs = dualmat.rows();
  int perm_count = 0;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      base_permutations(perm_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(wcoeffs, dualmat);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t i = 0; i < topology.size(); ++i)
    entity_dofs[i].resize(topology[i].size(), 0);
  for (int& q : entity_dofs[1])
    q = degree + 1;
  for (int& q : entity_dofs[2])
    q = 3 * (degree + 1) * degree / 2;
  if (tdim > 2)
    entity_dofs[3] = {(degree + 1) * degree * (degree - 1)};

  FiniteElement el(celltype, degree, {tdim, tdim}, new_coeffs, entity_dofs,
                   base_permutations);
  return el;
}
//-----------------------------------------------------------------------------
