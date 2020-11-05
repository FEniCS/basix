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

FiniteElement RaviartThomas::create(cell::Type celltype, int degree)
{
  if (celltype != cell::Type::triangle and celltype != cell::Type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const int tdim = cell::topological_dimension(celltype);

  // The number of order (degree-1) scalar polynomials
  const int nv = polyset::size(celltype, degree - 1);
  // The number of order (degree-2) scalar polynomials
  const int ns0 = polyset::size(celltype, degree - 2);
  // The number of additional polnomials in the polynomial basis for
  // Raviart-Thomas
  const int ns = (tdim == 2) ? degree : degree * (degree + 1) / 2;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, Qwts] = quadrature::make_quadrature(tdim, 2 * degree);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = polyset::tabulate(celltype, degree, 0, Qpts)[0];

  // The number of order (degree) polynomials
  const int psize = Pkp1_at_Qpts.cols();

  // Create coefficients for order (degree-1) vector polynomials
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs
      = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>::Zero(nv * tdim + ns, psize * tdim);
  for (int j = 0; j < tdim; ++j)
  {
    wcoeffs.block(nv * j, psize * j, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);
  }

  // Create coefficients for additional polynomials in Raviart-Thomas polynomial
  // basis
  for (int i = 0; i < ns; ++i)
  {
    for (int k = 0; k < psize; ++k)
    {
      for (int j = 0; j < tdim; ++j)
      {
        auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(j)
                 * Pkp1_at_Qpts.col(k);
        wcoeffs(nv * tdim + i, k + psize * j) = w.sum();
      }
    }
  }

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat
      = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>::Zero(nv * tdim + ns, psize * tdim);

  // quadrature degree
  int quad_deg = 5 * degree;

  // Create a polynomial set on a reference facet
  cell::Type facettype
      = (tdim == 2) ? cell::Type::interval : cell::Type::triangle;

  // Add rows to dualmat for integral moments on facets
  FiniteElement moment_space_facet
      = DiscontinuousLagrange::create(facettype, degree - 1);
  const int facet_count = tdim + 1;
  const int facet_dofs = (tdim == 2) ? degree : (degree * (degree + 1) / 2);
  dualmat.block(0, 0, facet_count * facet_dofs, psize * tdim)
      = moments::make_normal_integral_moments(moment_space_facet, celltype,
                                              tdim, degree, quad_deg);

  // Add rows to dualmat for integral moments on interior
  if (degree > 1)
  {
    const int internal_dofs = (tdim == 2)
                                  ? (degree * (degree - 1))
                                  : (degree * (degree - 1) * (degree + 1) / 2);
    // Interior integral moment
    FiniteElement moment_space_interior
        = DiscontinuousLagrange::create(celltype, degree - 2);
    dualmat.block(facet_count * facet_dofs, 0, internal_dofs, psize * tdim)
        = moments::make_integral_moments(moment_space_interior, celltype, tdim,
                                         degree, quad_deg);
  }

  // TODO
  const int ndofs = dualmat.rows();
  int perm_count = 0;
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      base_permutations(perm_count, ndofs);

  auto new_coeffs = FiniteElement::apply_dualmat_to_basis(wcoeffs, dualmat);
  FiniteElement el(celltype, degree, tdim, new_coeffs, base_permutations);
  return el;
}
//-----------------------------------------------------------------------------
