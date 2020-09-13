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

  // PkH subset
  const int ns = _degree + 1;
  std::vector<int> scalar_idx(ns);
  std::iota(scalar_idx.begin(), scalar_idx.end(), (_degree + 1) * _degree / 2);

  auto [Qpts, Qwts] = make_quadrature_triangle_collapsed(2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts(Pkp1.size(), Qpts.rows());
  for (std::size_t j = 0; j < Pkp1.size(); ++j)
    Pkp1_at_Qpts.row(j) = Pkp1[j].tabulate(Qpts);

  // Create coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2 + ns, Pkp1.size() * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, Pkp1.size(), nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (std::size_t k = 0; k < Pkp1.size(); ++k)
    {
      auto w0 = Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose() * Qpts.col(1)
                * Pkp1_at_Qpts.row(k).transpose();
      wcoeffs(2 * nv + i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose()
                * Qpts.col(0) * Pkp1_at_Qpts.row(k).transpose();
      wcoeffs(2 * nv + i, k + Pkp1.size()) = w1.sum();
    }

  std::cout << "Coeffs = \n[" << wcoeffs << "]\n";

  // Dual space

  // Iterate over edges
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      triangle_geom
      = triangle.reference_geometry();

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(nv * 2 + ns, nv * 4 + ns * 2);
  dualmat.setZero();

  // FIXME: need to add interior dofs for higher order

  // See FIAT dual_set to_riesz and functional

  // Get edge interior points, and tangent direction
  int c = 0;
  for (int i = 0; i < 3; ++i)
  {
    Eigen::Array<double, 2, 2, Eigen::RowMajor> edge;
    edge.row(0) = triangle_geom.row((i + 1) % 3);
    edge.row(1) = triangle_geom.row((i + 2) % 3);
    Eigen::Vector2d tangent = edge.row(1) - edge.row(0);
    std::cout << "tangent = " << tangent.transpose() << "\n";

    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        pts = ReferenceSimplex::make_lattice(_degree + 2, edge, false);
    std::cout << "pts = " << pts << "\n";

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        values(pts.rows(), Pkp1.size());
    for (std::size_t j = 0; j < Pkp1.size(); ++j)
      values.col(j) = Pkp1[j].tabulate(pts);
    std::cout << "values = " << values << "\n";

    for (int j = 0; j < pts.rows(); ++j)
    {
      for (int k = 0; k < nv * 2 + ns; ++k)
      {
        dualmat(c, k) = tangent[0] * values(j, k);
        dualmat(c, k + nv * 2 + ns) = tangent[1] * values(j, k);
      }
      ++c;
    }
  }
  std::cout << "dualmat = " << dualmat << "\n";

  // See FIAT in finite_element.py constructor
  auto A = wcoeffs * dualmat.transpose();
  auto Ainv = A.inverse();
  auto new_coeffs = Ainv * wcoeffs;
  std::cout << "new_coeffs = \n" << new_coeffs << "\n";

  // Create polynomial sets for x and y components
  // stacking x0, x1, x2,... y0, y1, y2,...
  poly_set.resize(nv * 4 + ns * 2, Polynomial::zero(2));
  for (int i = 0; i < nv * 2 + ns; ++i)
  {
    for (std::size_t j = 0; j < Pkp1.size(); ++j)
    {
      poly_set[i] += Pkp1[j] * new_coeffs(i, j);
      poly_set[i + nv * 2 + ns] += Pkp1[j] * new_coeffs(i, j + Pkp1.size());
    }
  }
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
