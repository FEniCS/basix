// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "quadrature.h"
#include "simplex.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <numeric>
#include <vector>

Nedelec2D::Nedelec2D(int k) : FiniteElement(2, k - 1)
{
  // Reference triangle
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> triangle
      = ReferenceSimplex::create_simplex(_dim);

  // Create orthonormal basis on triangle
  std::vector<Polynomial> Pkp1
      = ReferenceSimplex::compute_polynomial_set(_dim, _degree + 1);
  int psize = Pkp1.size();

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) / 2;

  // PkH subset
  const int ns = _degree + 1;
  const int ns0 = (_degree + 1) * _degree / 2;

  auto [Qpts, Qwts] = make_quadrature(_dim, 2 * _degree + 2);
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
      auto w0 = Qwts * Pkp1_at_Qpts.row(ns0 + i).transpose() * Qpts.col(1)
                * Pkp1_at_Qpts.row(k).transpose();
      wcoeffs(2 * nv + i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.row(ns0 + i).transpose() * Qpts.col(0)
                * Pkp1_at_Qpts.row(k).transpose();
      wcoeffs(2 * nv + i, k + psize) = w1.sum();
    }

  std::cout << "Initial coeffs = \n[" << wcoeffs << "]\n";

  // Dual space

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(nv * 2 + ns, psize * 2);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Integral representation for the boundary (edge) dofs

  // Create a polynomial set on a reference edge
  std::vector<Polynomial> Pq
      = ReferenceSimplex::compute_polynomial_set(1, _degree);

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (_degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(1, quad_deg);

  // Iterate over edges
  for (int i = 0; i < 3; ++i)
  {
    // FIXME: get tangent from the simplex class
    Eigen::Array<double, 2, 2, Eigen::RowMajor> edge
        = ReferenceSimplex::sub(triangle, 1, i);
    Eigen::Vector2d tangent = edge.row(1) - edge.row(0);

    // UFC convention?
    if (i == 1)
      tangent *= -1;

    // Map quadrature points onto triangle edge
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        QptsE_scaled(QptsE.rows(), 2);
    for (int j = 0; j < QptsE.rows(); ++j)
      QptsE_scaled.row(j)
          = edge.row(0) + QptsE(j, 0) * (edge.row(1) - edge.row(0));

    // Tabulate Pkp1 at edge quadrature points
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsE(psize, QptsE_scaled.rows());
    for (int j = 0; j < psize; ++j)
      Pkp1_at_QptsE.row(j) = Pkp1[j].tabulate(QptsE_scaled);

    // Compute edge tangent integral moments
    for (std::size_t j = 0; j < Pq.size(); ++j)
    {
      Eigen::ArrayXd phi = Pq[j].tabulate(QptsE);
      for (int k = 0; k < 2; ++k)
      {
        Eigen::VectorXd q = phi * QwtsE * tangent[k];
        Eigen::RowVectorXd qcoeffs = Pkp1_at_QptsE * q;
        dualmat.block(c, psize * k, 1, psize) = qcoeffs;
      }
      ++c;
    }
  }

  if (_degree > 0)
  {
    // Interior integral moment
    std::vector<Polynomial> Pkm1
        = ReferenceSimplex::compute_polynomial_set(2, _degree - 1);
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

  apply_dualmat_to_basis(wcoeffs, dualmat, Pkp1, _dim);
}
//-----------------------------------------------------------------------------
Nedelec3D::Nedelec3D(int k) : FiniteElement(3, k - 1)
{
  // Reference tetrahedron
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> simplex
      = ReferenceSimplex::create_simplex(_dim);

  // Create orthonormal basis on simplex
  std::vector<Polynomial> Pkp1
      = ReferenceSimplex::compute_polynomial_set(_dim, _degree + 1);
  int psize = Pkp1.size();

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) * (_degree + 3) / 6;

  // PkH subset
  const int ns = (_degree + 1) * (_degree + 2) / 2;
  const int ns0 = _degree * (_degree + 1) * (_degree + 2) / 6;

  auto [Qpts, Qwts] = make_quadrature(_dim, 2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts(psize, Qpts.rows());
  for (int j = 0; j < psize; ++j)
    Pkp1_at_Qpts.row(j) = Pkp1[j].tabulate(Qpts);

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * _dim + ns * _dim, psize * _dim);
  wcoeffs.setZero();
  for (int i = 0; i < _dim; ++i)
    wcoeffs.block(nv * i, psize * i, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (int k = 0; k < psize; ++k)
      for (int j = 0; j < _dim; ++j)
      {
        const int j1 = (j + 1) % 3;
        const int j2 = (j + 2) % 3;

        auto w = Qwts * Pkp1_at_Qpts.row(ns0 + i).transpose() * Qpts.col(j)
                 * Pkp1_at_Qpts.row(k).transpose();
        wcoeffs(_dim * nv + i + ns * j1, psize * j2 + k) = -w.sum();
        wcoeffs(_dim * nv + i + ns * j2, psize * j1 + k) = w.sum();
      }

  // Remove dependent components from space with SVD
  Eigen::JacobiSVD<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd(wcoeffs, Eigen::ComputeThinU | Eigen::ComputeThinV);

  int ndofs = 6 * (_degree + 1) + 4 * _degree * (_degree + 1)
              + (_degree - 1) * _degree * (_degree + 1) / 2;
  wcoeffs = svd.matrixV().transpose().topRows(ndofs);

  // Check singular values
  Eigen::VectorXd s = svd.singularValues();
  for (int i = 0; i < ndofs; ++i)
    if (s[i] < 1e-12)
      throw std::runtime_error("Error in Nedelec3D space");
  for (int i = ndofs; i < s.size(); ++i)
    if (s[i] > 1e-12)
      throw std::runtime_error("Error in Nedelec3D space");

  wcoeffs = (wcoeffs.array().abs() < 1e-16).select(0.0, wcoeffs);

  std::cout << "Initial coeffs = \n[" << wcoeffs << "]\n";

  // Dual space

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * _dim);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Integral representation for the boundary dofs

  // Create a polynomial set on a reference edge
  std::vector<Polynomial> Pq
      = ReferenceSimplex::compute_polynomial_set(1, _degree);

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (_degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(1, quad_deg);

  // Iterate over edges
  for (int i = 0; i < 6; ++i)
  {
    // FIXME: get tangent from the simplex class
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> edge
        = ReferenceSimplex::sub(simplex, 1, i);
    Eigen::Vector3d tangent = edge.row(1) - edge.row(0);
    std::cout << "Edge " << i << " " << tangent.transpose() << "\n";

    // Map quadrature points onto tetrahedron edge
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        QptsE_scaled(QptsE.rows(), _dim);
    for (int j = 0; j < QptsE.rows(); ++j)
      QptsE_scaled.row(j)
          = edge.row(0) + QptsE(j, 0) * (edge.row(1) - edge.row(0));

    // Tabulate Pkp1 at edge quadrature points
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsE(psize, QptsE_scaled.rows());
    for (int j = 0; j < psize; ++j)
      Pkp1_at_QptsE.row(j) = Pkp1[j].tabulate(QptsE_scaled);

    // Compute edge tangent integral moments
    for (std::size_t j = 0; j < Pq.size(); ++j)
    {
      Eigen::ArrayXd phi = Pq[j].tabulate(QptsE);
      for (int k = 0; k < 3; ++k)
      {
        Eigen::VectorXd q = phi * QwtsE * tangent[k];
        Eigen::RowVectorXd qcoeffs = Pkp1_at_QptsE * q;
        dualmat.block(c, psize * k, 1, psize) = qcoeffs;
      }
      ++c;
    }
  }

  if (_degree > 0)
  {
    // Create a polynomial set on a reference facet
    std::vector<Polynomial> PqF
        = ReferenceSimplex::compute_polynomial_set(2, _degree - 1);

    // Create quadrature scheme on the facet
    int quad_deg = 5 * (_degree + 1);
    auto [QptsF, QwtsF] = make_quadrature(2, quad_deg);

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          facet = ReferenceSimplex::sub(simplex, 2, i);
      // Face tangents
      Eigen::Vector3d t0 = facet.row(1) - facet.row(0);
      Eigen::Vector3d t1 = facet.row(2) - facet.row(0);

      // Map quadrature points onto tetrahedron facet
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          QptsF_scaled(QptsF.rows(), _dim);
      for (int j = 0; j < QptsF.rows(); ++j)
        QptsF_scaled.row(j) = facet.row(0)
                              + QptsF(j, 0) * (facet.row(1) - facet.row(0))
                              + QptsF(j, 1) * (facet.row(2) - facet.row(0));

      // Tabulate Pkp1 at facet quadrature points
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          Pkp1_at_QptsF(psize, QptsF_scaled.rows());
      for (int j = 0; j < psize; ++j)
        Pkp1_at_QptsF.row(j) = Pkp1[j].tabulate(QptsF_scaled);

      for (std::size_t j = 0; j < PqF.size(); ++j)
      {
        for (int k = 0; k < _dim; ++k)
        {
          // FIXME: check this over
          Eigen::ArrayXd phi = PqF[j].tabulate(QptsF);
          Eigen::VectorXd q0 = phi * QwtsF * t0[k];
          Eigen::RowVectorXd qcoeffs0 = Pkp1_at_QptsF * q0;
          Eigen::VectorXd q1 = phi * QwtsF * t1[k];
          Eigen::RowVectorXd qcoeffs1 = Pkp1_at_QptsF * q1;
          dualmat.block(c, psize * k, 1, psize) = qcoeffs0;
          dualmat.block(c + 1, psize * k, 1, psize) = qcoeffs1;
        }
        c += 2;
      }
    }
  }
  if (_degree > 1)
  {
    // Interior integral moment
    std::vector<Polynomial> Pkm2
        = ReferenceSimplex::compute_polynomial_set(_dim, _degree - 2);
    for (std::size_t i = 0; i < Pkm2.size(); ++i)
    {
      Eigen::ArrayXd phi = Pkm2[i].tabulate(Qpts);
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

  apply_dualmat_to_basis(wcoeffs, dualmat, Pkp1, _dim);
}
//-----------------------------------------------------------------------------
