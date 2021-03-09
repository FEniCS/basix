// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "regge.h"
#include "element-families.h"
#include "lattice.h"
#include "mappings.h"
#include "polyset.h"
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
Eigen::MatrixXd create_regge_space(cell::type celltype, int degree)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported celltype");

  const int tdim = cell::topological_dimension(celltype);
  const int nc = tdim * (tdim + 1) / 2;
  const int basis_size = polyset::dim(celltype, degree);
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
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd>
create_regge_interpolation(cell::type celltype, int degree)
{
  const std::size_t tdim = cell::topological_dimension(celltype);

  const int basis_size = polyset::dim(celltype, degree);

  const int ndofs = basis_size * (tdim + 1) * tdim / 2;
  const int space_size = basis_size * tdim * tdim;

  const std::size_t npoints
      = tdim == 2 ? 3 * (degree + 1) + degree * (degree + 1) / 2
                  : 6 * (degree + 1) + 4 * degree * (degree + 1) / 2
                        + degree * (degree + 1) * (degree - 1) / 6;

  xt::xtensor<double, 2> points({npoints, tdim});
  Eigen::ArrayXXd matrix(ndofs, npoints * tdim * tdim);
  matrix.setZero();

  Eigen::ArrayXXd dualmat(ndofs, space_size);
  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);

  // point and dof counters
  int point_n = 0;
  int dof = 0;
  for (std::size_t dim = 1; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const xt::xtensor<double, 2> entity_geom
          = cell::sub_entity_geometry(celltype, dim, i);

      // Eigen::ArrayXd point = entity_geom.row(0);
      cell::type ct = cell::sub_entity_type(celltype, dim, i);
      auto lattice
          = lattice::create(ct, degree + 2, lattice::type::equispaced, false);
      for (std::size_t j = 0; j < lattice.shape()[0]; ++j)
      {
        xt::row(points, point_n + j) = xt::row(entity_geom, 0);
        for (std::size_t k = 0; k < entity_geom.shape()[0] - 1; ++k)
        {
          xt::row(points, point_n + j)
              += (xt::row(entity_geom, k + 1) - xt::row(entity_geom, 0))
                 * lattice(j, k);
        }
      }

      Eigen::Map<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          _pt(points.data(), points.shape()[0], points.shape()[1]);

      Eigen::MatrixXd basis = polyset::tabulate(
          celltype, degree, 0,
          _pt.block(point_n, 0, lattice.shape()[0], tdim))[0];

      // Store up outer(t, t) for all tangents
      std::vector<int>& vert_ids = topology[dim][i];
      int ntangents = dim * (dim + 1) / 2;
      std::vector<Eigen::MatrixXd> vvt(ntangents);
      int c = 0;
      for (std::size_t s = 0; s < dim; ++s)
      {
        for (std::size_t d = s + 1; d < dim + 1; ++d)
        {
          Eigen::VectorXd edge_t(geometry.shape()[1]);
          for (std::size_t p = 0; p < geometry.shape()[1]; ++p)
            edge_t[p] = geometry(vert_ids[d], p) - geometry(vert_ids[s], p);
          // const Eigen::VectorXd edge_t
          //     = geometry.row(vert_ids[d]) - geometry.row(vert_ids[s]);

          // outer product v.v^T
          vvt[c++] = edge_t * edge_t.transpose();
        }
      }

      for (std::size_t k = 0; k < lattice.shape()[0]; ++k)
      {
        for (int j = 0; j < ntangents; ++j)
        {
          Eigen::Map<Eigen::VectorXd> vvt_flat(vvt[j].data(),
                                               vvt[j].rows() * vvt[j].cols());
          for (std::size_t i = 0; i < tdim * tdim; ++i)
            matrix(dof, point_n + i * npoints) = vvt_flat(i);
          Eigen::Map<Eigen::RowVectorXd>(vvt[j].data(),
                                         vvt[j].rows() * vvt[j].cols());
          ++dof;
        }
        ++point_n;
      }
    }
  }

  Eigen::Map<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      _pt(points.data(), points.shape()[0], points.shape()[1]);
  return std::make_pair(_pt, matrix);
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
FiniteElement basix::create_regge(cell::type celltype, int degree)
{
  const int tdim = cell::topological_dimension(celltype);
  const int basis_size = polyset::dim(celltype, degree);
  const int ndofs = basis_size * (tdim + 1) * tdim / 2;

  Eigen::MatrixXd wcoeffs = create_regge_space(celltype, degree);

  Eigen::ArrayXXd points;
  Eigen::MatrixXd matrix;

  std::tie(points, matrix) = create_regge_interpolation(celltype, degree);

  // TODO

  int transform_count = tdim == 2 ? 3 : 14;
  std::vector<Eigen::MatrixXd> base_transformations(
      transform_count, Eigen::MatrixXd::Identity(ndofs, ndofs));

  Eigen::MatrixXd coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, matrix, points, degree);

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

  return FiniteElement(element::family::Regge, celltype, degree, {tdim, tdim},
                       coeffs, entity_dofs, base_transformations, points,
                       matrix, mapping::type::doubleCovariantPiola);
}
//-----------------------------------------------------------------------------
