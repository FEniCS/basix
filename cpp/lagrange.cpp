// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lagrange.h"
#include "polynomial-set.h"
#include <Eigen/Dense>
#include <iostream>

Lagrange::Lagrange(Cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != Cell::Type::interval and celltype != Cell::Type::triangle
      and celltype != Cell::Type::tetrahedron)
    throw std::runtime_error("Invalid celltype");

  // Create points at nodes
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = Cell::create_lattice(celltype, degree, true);
  int ndofs = pt.rows();

  std::vector<std::vector<std::vector<int>>> topology
      = Cell::topology(celltype);
  auto geometry = Cell::geometry(celltype);
  int c = 0;
  for (std::size_t dim = 0; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          entity_geom(topology[dim][i].size(), geometry.cols());
      for (std::size_t j = 0; j < topology[dim][i].size(); ++j)
        entity_geom.row(j) = geometry.row(topology[dim][i][j]);

      Eigen::ArrayXd point = entity_geom.row(0);
      Cell::Type ct = Cell::simplex_type(dim);

      auto lattice = Cell::create_lattice(ct, degree, false);
      for (int j = 0; j < lattice.rows(); ++j)
      {
        pt.row(c) = entity_geom.row(0);
        for (int k = 0; k < entity_geom.rows() - 1; ++k)
          pt.row(c)
              += (entity_geom.row(k + 1) - entity_geom.row(0)) * lattice(j, k);

        std::cout << "Add point [" << point << "]\n";
        ++c;
      }
    }
  }
  std::cout << "Added " << c << "points \n";

  // Coefficients are Identity Matrix
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);

  // Point evaluation of basis
  Eigen::MatrixXd dualmat
      = PolynomialSet::tabulate_polynomial_set(celltype, degree, pt);

  apply_dualmat_to_basis(coeffs, dualmat);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Lagrange::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      basis_at_pts
      = PolynomialSet::tabulate_polynomial_set(_cell_type, _degree, pts);

  return basis_at_pts * _coeffs.transpose();
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Lagrange::tabulate_basis_derivatives(
    int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dbasis_at_pts = PolynomialSet::tabulate_polynomial_set_deriv(
          _cell_type, _degree, nderiv, pts);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(dbasis_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
    dresult[p] = dbasis_at_pts[p].matrix() * _coeffs.transpose();

  return dresult;
}
