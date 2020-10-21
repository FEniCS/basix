// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "polynomial-set.h"
#include <Eigen/Dense>
#include <iostream>

using namespace libtab;

Lagrange::Lagrange(Cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != Cell::Type::interval and celltype != Cell::Type::triangle
      and celltype != Cell::Type::tetrahedron)
    throw std::runtime_error("Invalid celltype");

  const int ndofs = PolynomialSet::size(celltype, degree);
  const int tdim = Cell::topological_dimension(celltype);

  // Create points at nodes, ordered by topology (vertices first)
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt(
      ndofs, tdim);

  std::vector<std::vector<std::vector<int>>> topology
      = Cell::topology(celltype);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geometry
      = Cell::geometry(celltype);
  int c = 0;
  for (std::size_t dim = 0; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          entity_geom = Cell::sub_entity_geometry(celltype, dim, i);

      Eigen::ArrayXd point = entity_geom.row(0);
      Cell::Type ct = Cell::simplex_type(dim);

      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          lattice = Cell::create_lattice(ct, degree, false);
      for (int j = 0; j < lattice.rows(); ++j)
      {
        pt.row(c) = entity_geom.row(0);
        for (int k = 0; k < entity_geom.rows() - 1; ++k)
          pt.row(c)
              += (entity_geom.row(k + 1) - entity_geom.row(0)) * lattice(j, k);
        ++c;
      }
    }
  }

  // Initial coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // Point evaluation of basis
  Eigen::MatrixXd dualmat = PolynomialSet::tabulate(celltype, degree, 0, pt)[0];

  apply_dualmat_to_basis(coeffs, dualmat);
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Lagrange::tabulate(int nderiv,
                   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dbasis_at_pts = PolynomialSet::tabulate(_cell_type, _degree, nderiv, pts);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(dbasis_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
    dresult[p] = dbasis_at_pts[p].matrix() * _coeffs.transpose();

  return dresult;
}
