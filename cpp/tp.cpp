// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "tp.h"
#include "polynomial-set.h"
#include <Eigen/Dense>

using namespace libtab;

FiniteElement TensorProduct::create(cell::Type celltype, int degree)
{
  if (celltype != cell::Type::quadrilateral and celltype != cell::Type::prism
      and celltype != cell::Type::pyramid
      and celltype != cell::Type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  // Iterate through topology
  std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const int tdim = cell::topological_dimension(celltype);
  const int ndofs = polyset::size(celltype, degree);

  std::vector<std::vector<int>> entity_dofs(tdim + 1);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt(
      ndofs, tdim);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geometry
      = cell::geometry(celltype);
  int c = 0;
  for (std::size_t dim = 0; dim < topology.size(); ++dim)
  {
    for (std::size_t i = 0; i < topology[dim].size(); ++i)
    {
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          entity_geom = cell::sub_entity_geometry(celltype, dim, i);

      Eigen::ArrayXd point = entity_geom.row(0);
      if (dim == 0)
      {
        pt.row(c) = entity_geom.row(0);
        entity_dofs[0].push_back(1);
        ++c;
      }
      else
      {
        cell::Type ct = cell::sub_entity_type(celltype, dim, i);
        const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
            lattice = cell::create_lattice(ct, degree, false);
        entity_dofs[dim].push_back(lattice.rows());
        for (int j = 0; j < lattice.rows(); ++j)
        {
          pt.row(c) = entity_geom.row(0);
          for (int k = 0; k < lattice.cols(); ++k)
          {
            pt.row(c) += (entity_geom.row(k + 1) - entity_geom.row(0))
                         * lattice(j, k);
          }
          ++c;
        }
      }
    }
  }

  Eigen::MatrixXd dualmat = polyset::tabulate(celltype, degree, 0, pt)[0];
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(ndofs, ndofs);
  auto new_coeffs
      = FiniteElement::compute_expansion_coefficents(coeffs, dualmat);
  FiniteElement el(celltype, degree, 1, new_coeffs, entity_dofs);
  return el;
}
