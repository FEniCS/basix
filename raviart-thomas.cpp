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
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> simplex
      = ReferenceSimplex::create_simplex(dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> Pkp1
      = ReferenceSimplex::compute_polynomial_set(_dim, _degree + 1);
  int psize = Pkp1.size();

  // Vector subsets
  int nv;
  int ns0;
  int ns;
  if (_dim == 2)
  {
    nv = (_degree + 1) * (_degree + 2) / 2;
    ns0 = _degree * (_degree + 1) / 2;
    ns = (_degree + 1);
  }
  else
  {
    assert(_dim == 3);
    nv = (_degree + 1) * (_degree + 2) * (_degree + 3) / 6;
    ns0 = _degree * (_degree + 1) * (_degree + 2) / 6;
    ns = (_degree + 1) * (_degree + 2) / 2;
  }

  std::cout << "nv = " << nv << "\n";
  std::cout << "ns = " << ns << "\n";
  std::cout << "ns0 = " << ns0 << "\n";

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
        auto w = Qwts * Pkp1_at_Qpts.row(ns0 + i).transpose() * Qpts.col(j)
                 * Pkp1_at_Qpts.row(k).transpose();
        wcoeffs(nv * _dim + i, k + psize * j) = w.sum();
      }

  std::cout << "Initial coeffs = \n[" << wcoeffs << "]\n";

  // Dual space

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(nv * _dim + ns, psize * _dim);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Create a polynomial set on a reference facet
  std::vector<Polynomial> Pq
      = ReferenceSimplex::compute_polynomial_set(_dim - 1, _degree);
  // Create quadrature scheme on the facet
  int quad_deg = 5 * (_degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(_dim - 1, quad_deg);

  for (int i = 0; i < (_dim + 1); ++i)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> facet
        = ReferenceSimplex::sub(simplex, _dim - 1, i);

    // FIXME: get normal from the simplex class
    Eigen::VectorXd normal;
    if (_dim == 2)
    {
      normal.resize(2);
      normal << facet(1, 1) - facet(0, 1), facet(0, 0) - facet(1, 0);
      if (i == 1)
        normal *= -1;
    }
    else if (_dim == 3)
    {
      Eigen::Vector3d e0 = facet.row(1) - facet.row(0);
      Eigen::Vector3d e1 = facet.row(2) - facet.row(0);
      normal = e1.cross(e0);
    }

    // Map reference facet to cell
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        QptsE_scaled(QptsE.rows(), _dim);
    for (int j = 0; j < QptsE.rows(); ++j)
    {
      QptsE_scaled.row(j) = facet.row(0);
      for (int k = 0; k < (_dim - 1); ++k)
        QptsE_scaled.row(j) += QptsE(j, k) * (facet.row(k + 1) - facet.row(0));
    }

    // Tabulate Pkp1 at facet quadrature points
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsE(psize, QptsE_scaled.rows());
    for (int j = 0; j < psize; ++j)
      Pkp1_at_QptsE.row(j) = Pkp1[j].tabulate(QptsE_scaled);

    // Compute facet normal integral moments by quadrature
    for (std::size_t j = 0; j < Pq.size(); ++j)
    {
      Eigen::ArrayXd phi = Pq[j].tabulate(QptsE);
      for (int k = 0; k < _dim; ++k)
      {
        Eigen::VectorXd q = phi * QwtsE * normal[k];
        Eigen::RowVectorXd qcoeffs = Pkp1_at_QptsE.matrix() * q;
        dualmat.block(c, psize * k, 1, psize) = qcoeffs;
      }
      ++c;
    }
  }

  // Should work for 2D and 3D
  if (_degree > 0)
  {
    // Interior integral moment
    std::vector<Polynomial> Pkm1
        = ReferenceSimplex::compute_polynomial_set(_dim, _degree - 1);
    for (std::size_t i = 0; i < Pkm1.size(); ++i)
    {
      Eigen::ArrayXd phi = Pkm1[i].tabulate(Qpts);
      Eigen::VectorXd q = phi * Qwts;
      Eigen::RowVectorXd qcoeffs = Pkp1_at_Qpts.matrix() * q;
      assert(qcoeffs.size() == psize);
      for (int j = 0; j < _dim; ++j)
      {
        dualmat.block(c, psize * j, 1, psize) = qcoeffs;
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
