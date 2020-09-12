// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "quadrature.h"
#include "simplex.h"
#include <Eigen/SVD>
#include <numeric>

Nedelec2D::Nedelec2D(int k) : _dim(2), _degree(k - 1)
{
  // Reference triangle
  ReferenceSimplex triangle(2);

  // Create orthonormal basis on triangle
  std::vector<Polynomial> Pkp1 = triangle.compute_polynomial_set(_degree + 1);

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) / 2;
  std::vector<int> vec_idx(nv);
  std::iota(vec_idx.begin(), vec_idx.end(), 0);

  // PkH subset
  const int ns = _degree + 1;
  std::vector<int> scalar_idx(ns);
  std::iota(scalar_idx.begin(), scalar_idx.end(), (_degree + 1) * _degree / 2);

  auto [Qpts, Qwts] = make_quadrature_triangle_collapsed(2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts(Pkp1.size(), Qpts.rows());
  for (std::size_t j = 0; j < Pkp1.size(); ++j)
    Pkp1_at_Qpts.row(j) = Pkp1[j].tabulate(Qpts);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      PkH_crossx_coeffs_0(ns, Pkp1.size());
  PkH_crossx_coeffs_0.setZero();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      PkH_crossx_coeffs_1(ns, Pkp1.size());
  PkH_crossx_coeffs_1.setZero();

  for (int i = 0; i < ns; ++i)
    for (std::size_t k = 0; k < Pkp1.size(); ++k)
    {
      auto w0 = Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose() * Qpts.col(1)
                * Pkp1_at_Qpts.row(k).transpose();
      PkH_crossx_coeffs_0(i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose()
                * Qpts.col(0) * Pkp1_at_Qpts.row(k).transpose();
      PkH_crossx_coeffs_1(i, k) = w1.sum();
    }

  // Create polynomial sets for x and y components
  std::vector<Polynomial> poly_set_x(nv * 2 + ns, Polynomial::zero(2));
  std::vector<Polynomial> poly_set_y(nv * 2 + ns, Polynomial::zero(2));
  for (int i = 0; i < nv; ++i)
  {
    poly_set_x[i] = Pkp1[i];
    poly_set_y[i + nv] = Pkp1[i];
  }
  for (int i = 0; i < ns; ++i)
  {
    for (std::size_t j = 0; j < Pkp1.size(); ++j)
    {
      poly_set_x[i + 2 * nv] += Pkp1[j] * PkH_crossx_coeffs_0(i, j);
      poly_set_y[i + 2 * nv] += Pkp1[j] * PkH_crossx_coeffs_1(i, j);
    }
  }

  poly_set.resize(nv * 4 + ns * 2);
  for (int i = 0; i < nv * 2 + ns; ++i)
  {
    poly_set[i] = poly_set_x[i];
    poly_set[i + nv * 2 + ns] = poly_set_y[i];
  }

  // Dual space
  ReferenceSimplex edge(1);
  int quad_deg = 2; //? FIXME
  auto [QptsL, QwtsL] = make_quadrature_line(quad_deg);
  std::vector<Polynomial> Pq
      = edge.compute_polynomial_set(_degree); //? FIXME - check degree

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pq_at_qpts(QptsL.rows(), Pq.size());
  for (std::size_t j = 0; j < Pq.size(); ++j)
    Pq_at_qpts.col(j) = Pq[j].tabulate(QptsL);

  //            for e in range(len(t[sd - 1])):
  //                for i in range(Pq_at_qpts.shape[0]):
  //                    phi = Pq_at_qpts[i, :]
  //                    nodes.append(functional.IntegralMomentOfEdgeTangentEvaluation(ref_el,
  //                    Q, phi, e))
}
//-----------------------------------------------------------------------------
// Compute basis values at set of points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Nedelec2D::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  if (pts.cols() != _dim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), poly_set.size());
  for (std::size_t j = 0; j < poly_set.size(); ++j)
    result.col(j) = poly_set[j].tabulate(pts);

  return result;
}
