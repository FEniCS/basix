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
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> triangle
      = ReferenceSimplex::create_simplex(2);

  // Create orthonormal basis on triangle
  std::vector<Polynomial> Pkp1
      = ReferenceSimplex::compute_polynomial_set(triangle, _degree + 1);
  int psize = Pkp1.size();

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) / 2;

  // PkH subset
  const int ns = _degree + 1;
  std::vector<int> scalar_idx(ns);
  std::iota(scalar_idx.begin(), scalar_idx.end(), (_degree + 1) * _degree / 2);

  auto [Qpts, Qwts] = make_quadrature_triangle_collapsed(2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts(psize, Qpts.rows());
  for (int j = 0; j < psize; ++j)
    Pkp1_at_Qpts.row(j) = Pkp1[j].tabulate(Qpts);

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2 + ns, psize * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, psize, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (int k = 0; k < psize; ++k)
    {
      auto w0 = Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose() * Qpts.col(1)
                * Pkp1_at_Qpts.row(k).transpose();
      wcoeffs(2 * nv + i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose()
                * Qpts.col(0) * Pkp1_at_Qpts.row(k).transpose();
      wcoeffs(2 * nv + i, k + psize) = w1.sum();
    }

  std::cout << "Initial coeffs = \n[" << wcoeffs << "]\n";

  // Dual space

  // Iterate over edges

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(nv * 2 + ns, psize * 2);
  dualmat.setZero();

  // See FIAT nedelec, dual_set.to_riesz and functional

  // Get edge interior points, and tangent direction
  int c = 0;
  for (int i = 0; i < 3; ++i)
  {
    // FIXME: get this from the simplex class
    // FIXME: using Point Tangent evaluation - should use integral moment?
    Eigen::Array<double, 2, 2, Eigen::RowMajor> edge;
    edge.row(0) = triangle.row((i + 1) % 3);
    edge.row(1) = triangle.row((i + 2) % 3);
    Eigen::Vector2d tangent = edge.row(1) - edge.row(0);

    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        pts = ReferenceSimplex::create_lattice(edge, _degree + 2, false);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        values(pts.rows(), psize);
    for (int j = 0; j < psize; ++j)
      values.col(j) = Pkp1[j].tabulate(pts);

    for (int j = 0; j < pts.rows(); ++j)
    {
      for (int k = 0; k < psize; ++k)
      {
        dualmat(c, k) = tangent[0] * values(j, k);
        dualmat(c, k + psize) = tangent[1] * values(j, k);
      }
      ++c;
    }
  }

  if (_degree > 0)
  {
    // Interior integral moment
    std::vector<Polynomial> Pkm1
        = ReferenceSimplex::compute_polynomial_set(triangle, _degree - 1);
    for (std::size_t i = 0; i < Pkm1.size(); ++i)
    {
      Eigen::ArrayXd phi = Pkm1[i].tabulate(Qpts);
      Eigen::VectorXd q = phi * Qwts;
      Eigen::RowVectorXd qcoeffs = Pkp1_at_Qpts.matrix() * q;
      assert(qcoeffs.size() == psize);
      std::cout << "q = [" << q.transpose() << "]\n";
      dualmat.block(c, 0, 1, psize) = qcoeffs;
      ++c;
      dualmat.block(c, psize, 1, psize) = qcoeffs;
      ++c;
    }
  }

  std::cout << "dualmat = \n[" << dualmat << "]\n";

  // See FIAT in finite_element.py constructor
  auto A = wcoeffs * dualmat.transpose();
  auto Ainv = A.inverse();
  auto new_coeffs = Ainv * wcoeffs;
  std::cout << "new_coeffs = \n[" << new_coeffs << "]\n";

  // Create polynomial sets for x and y components
  // stacking x0, x1, x2,... y0, y1, y2,...
  poly_set.resize((nv * 2 + ns) * _dim, Polynomial::zero(2));
  for (int j = 0; j < _dim; ++j)
    for (int i = 0; i < nv * 2 + ns; ++i)
      for (int k = 0; k < psize; ++k)
        poly_set[i + (nv * 2 + ns) * j]
            += Pkp1[k] * new_coeffs(i, k + psize * j);
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
