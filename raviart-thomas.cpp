// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "raviart-thomas.h"
#include "quadrature.h"
#include "simplex.h"
#include <Eigen/SVD>
#include <numeric>

RaviartThomas::RaviartThomas(int dim, int k) : _dim(dim), _degree(k - 1)
{
  // Reference triangle
  ReferenceSimplex simplex(_dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> Pkp1 = simplex.compute_polynomial_set(_degree + 1);
  int psize = Pkp1.size();

  // Vector subset (2D or 3D? FIXME)
  int nv = (_degree + 1) * (_degree + 2) / 2;
  int ns0 = _degree * (_degree + 1) / 2;
  if (_dim == 3)
  {
    nv *= (_degree + 3) / 3;
    ns0 *= (_degree + 2) / 3;
  }

  // PkH subset = 2D ?3D FIXME?
  const int ns = _degree + _dim - 1;

  std::cout << "nv = " << nv << "\n";
  std::cout << "ns = " << ns << "\n";

  std::vector<int> scalar_idx(ns);
  std::iota(scalar_idx.begin(), scalar_idx.end(), ns0);

  auto [Qpts, Qwts] = make_quadrature(_dim, 2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts(psize, Qpts.rows());
  for (int j = 0; j < psize; ++j)
    Pkp1_at_Qpts.row(j) = Pkp1[j].tabulate(Qpts);

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * _dim + ns, psize * _dim);
  wcoeffs.setZero();
  for (int j = 0; j < _dim; ++j)
    wcoeffs.block(nv * j, psize * j, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (int k = 0; k < psize; ++k)
      for (int j = 0; j < _dim; ++j)
      {
        auto w = Qwts * Pkp1_at_Qpts.row(scalar_idx[i]).transpose()
                 * Qpts.col(j) * Pkp1_at_Qpts.row(k).transpose();
        wcoeffs(nv * _dim + i, k + psize * j) = w.sum();
      }

  std::cout << "Initial coeffs = \n[" << wcoeffs << "]\n";

  // Dual space

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(nv * _dim + ns, psize * _dim);
  dualmat.setZero();

  // 3D to do

  if (_dim == 2)
  {
    // Iterate over edges
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        triangle_geom
        = simplex.reference_geometry();

    // Get edge interior points, and tangent direction
    int c = 0;
    for (int i = 0; i < 3; ++i)
    {
      // FIXME: get this from simplex
      Eigen::Array<double, 2, 2, Eigen::RowMajor> edge;
      edge.row(0) = triangle_geom.row((i + 1) % 3);
      edge.row(1) = triangle_geom.row((i + 2) % 3);
      Eigen::Vector2d normal;
      normal << edge(1, 1) - edge(0, 1), edge(0, 0) - edge(1, 0);

      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          pts = ReferenceSimplex::make_lattice(_degree + 2, edge, false);

      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          values(pts.rows(), psize);
      for (int j = 0; j < psize; ++j)
        values.col(j) = Pkp1[j].tabulate(pts);

      for (int j = 0; j < pts.rows(); ++j)
      {
        for (int k = 0; k < psize; ++k)
        {
          dualmat(c, k) = normal[0] * values(j, k);
          dualmat(c, k + psize) = normal[1] * values(j, k);
        }
        ++c;
      }
    }

    if (_degree > 0)
    {
      // Interior integral moment
      std::vector<Polynomial> Pkm1
          = simplex.compute_polynomial_set(_degree - 1);
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
  }

  std::cout << "dualmat = \n[" << dualmat << "]\n";

  // See FIAT in finite_element.py constructor
  auto A = wcoeffs * dualmat.transpose();
  auto Ainv = A.inverse();
  auto new_coeffs = Ainv * wcoeffs;
  std::cout << "new_coeffs = \n[" << new_coeffs << "]\n";

  // Create polynomial sets for x and y components
  // stacking x0, x1, x2,... y0, y1, y2,...
  poly_set.resize((nv * _dim + ns) * _dim, Polynomial::zero(2));

  for (int j = 0; j < _dim; ++j)
    for (int i = 0; i < nv * _dim + ns; ++i)
      for (int k = 0; k < psize; ++k)
        poly_set[i + (nv * _dim + ns) * j]
            += Pkp1[k] * new_coeffs(i, k + psize * j);
}
//-----------------------------------------------------------------------------
// Compute basis values at set of points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
RaviartThomas::tabulate_basis(
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
