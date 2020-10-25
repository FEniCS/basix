// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "raviart-thomas.h"
#include "integral-moments.h"
#include "lagrange.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace libtab;

RaviartThomas::RaviartThomas(cell::Type celltype, int degree)
    : FiniteElement(celltype, degree)
{
  if (celltype != cell::Type::triangle and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const int tdim = cell::topological_dimension(celltype);
  this->_value_size = tdim;

  // Vector subsets
  const int nv = polyset::size(celltype, _degree - 1);
  const int ns0 = polyset::size(celltype, _degree - 2);
  const int ns = (tdim == 2) ? _degree : _degree * (_degree + 1) / 2;

  auto [Qpts, Qwts] = quadrature::make_quadrature(tdim, 2 * _degree);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = polyset::tabulate(celltype, _degree, 0, Qpts)[0];

  const int psize = Pkp1_at_Qpts.cols();

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * tdim + ns, psize * tdim);
  wcoeffs.setZero();
  for (int j = 0; j < tdim; ++j)
    wcoeffs.block(nv * j, psize * j, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (int k = 0; k < psize; ++k)
      for (int j = 0; j < tdim; ++j)
      {
        auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(j)
                 * Pkp1_at_Qpts.col(k);
        wcoeffs(nv * tdim + i, k + psize * j) = w.sum();
      }

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(nv * tdim + ns, psize * tdim);
  dualmat.setZero();

  // quadrature degree
  int quad_deg = 5 * _degree;

  // Create a polynomial set on a reference facet
  cell::Type facettype
      = (tdim == 2) ? cell::Type::interval : cell::Type::triangle;
  int facet_count = tdim + 1;

  Lagrange moment_space_facet(facettype, degree - 1);

  const int facet_dofs = (tdim == 2) ? _degree : (_degree * (_degree + 1) / 2);

  // Add integral moments on facets
  dualmat.block(0, 0, facet_count * facet_dofs, psize * tdim)
      = moments::make_normal_integral_moments(
          moment_space_facet, celltype, tdim, _degree, quad_deg);

  // Should work for 2D and 3D
  if (_degree > 1)
  {
    const int internal_dofs
        = (tdim == 2) ? (_degree * (_degree - 1))
                      : (_degree * (_degree - 1) * (_degree + 1) / 2);
    // Interior integral moment
    Lagrange moment_space_interior(celltype, degree - 2);
    dualmat.block(facet_count * facet_dofs, 0, internal_dofs, psize * tdim)
        = moments::make_integral_moments(
            moment_space_interior, celltype, tdim, _degree, quad_deg);
  }

  apply_dualmat_to_basis(wcoeffs, dualmat);
}
//-----------------------------------------------------------------------------
