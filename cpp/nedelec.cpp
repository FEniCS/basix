// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "nedelec.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <numeric>
#include <vector>

Nedelec2D::Nedelec2D(int k) : FiniteElement(Cell::Type::triangle, k - 1)
{
  // Reference triangle
  const int tdim = 2;
  Cell simplex_cell(Cell::Type::triangle);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> triangle
      = simplex_cell.geometry();

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) / 2;

  // PkH subset
  const int ns = _degree + 1;
  const int ns0 = (_degree + 1) * _degree / 2;
  const int ndofs = 3 * (_degree + 1) + _degree * (_degree + 1);
  assert(ndofs == nv * 2 + ns);

  auto [Qpts, Qwts] = make_quadrature(tdim, 2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
          Cell::Type::triangle, _degree + 1, Qpts);

  const int psize = Pkp1_at_Qpts.cols();

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs(nv * 2 + ns, psize * 2);
  wcoeffs.setZero();
  wcoeffs.block(0, 0, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);
  wcoeffs.block(nv, psize, nv, nv) = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (int k = 0; k < psize; ++k)
    {
      auto w0 = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(1)
                * Pkp1_at_Qpts.col(k);
      wcoeffs(2 * nv + i, k) = w0.sum();

      auto w1 = -Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(0)
                * Pkp1_at_Qpts.col(k);
      wcoeffs(2 * nv + i, k + psize) = w1.sum();
    }

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * 2);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Integral representation for the boundary (edge) dofs

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (_degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(1, quad_deg);

  // Tabulate a polynomial set on a reference edge
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pq_at_QptsE = PolynomialSet::tabulate_polynomial_set(Cell::Type::interval,
                                                           _degree, QptsE);
  // Iterate over edges
  for (int i = 0; i < 3; ++i)
  {
    // FIXME: get edge tangent from the cell class
    Eigen::Array<double, 2, 2, Eigen::RowMajor> edge
        = simplex_cell.sub_entity_geometry(1, i);
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
        Pkp1_at_QptsE = PolynomialSet::tabulate_polynomial_set(
                            Cell::Type::triangle, _degree + 1, QptsE_scaled)
                            .transpose();

    // Compute edge tangent integral moments
    for (int j = 0; j < Pq_at_QptsE.cols(); ++j)
    {
      Eigen::ArrayXd phi = Pq_at_QptsE.col(j);
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkm1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::triangle, _degree - 1, Qpts);
    for (int i = 0; i < Pkm1_at_Qpts.cols(); ++i)
    {
      Eigen::ArrayXd phi = Pkm1_at_Qpts.col(i);
      Eigen::VectorXd q = phi * Qwts;
      Eigen::RowVectorXd qcoeffs = Pkp1_at_Qpts.matrix().transpose() * q;
      assert(qcoeffs.size() == psize);
      dualmat.block(c, 0, 1, psize) = qcoeffs;
      ++c;
      dualmat.block(c, psize, 1, psize) = qcoeffs;
      ++c;
    }
  }

  apply_dualmat_to_basis(wcoeffs, dualmat);
}
//-----------------------------------------------------------------------------
Nedelec3D::Nedelec3D(int k) : FiniteElement(Cell::Type::tetrahedron, k - 1)
{
  // Reference tetrahedron
  const int tdim = 3;
  Cell simplex_cell(Cell::Type::tetrahedron);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> simplex
      = simplex_cell.geometry();

  // Vector subset
  const int nv = (_degree + 1) * (_degree + 2) * (_degree + 3) / 6;

  // PkH subset
  const int ns = (_degree + 1) * (_degree + 2) / 2;
  const int ns0 = _degree * (_degree + 1) * (_degree + 2) / 6;

  auto [Qpts, Qwts] = make_quadrature(tdim, 2 * _degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
          Cell::Type::tetrahedron, _degree + 1, Qpts);
  const int psize = Pkp1_at_Qpts.cols();

  // Create initial coefficients of Pkp1.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      wcoeffs((nv + ns) * tdim, psize * tdim);
  wcoeffs.setZero();
  for (int i = 0; i < tdim; ++i)
    wcoeffs.block(nv * i, psize * i, nv, nv)
        = Eigen::MatrixXd::Identity(nv, nv);

  for (int i = 0; i < ns; ++i)
    for (int k = 0; k < psize; ++k)
      for (int j = 0; j < tdim; ++j)
      {
        const int j1 = (j + 1) % 3;
        const int j2 = (j + 2) % 3;

        auto w = Qwts * Pkp1_at_Qpts.col(ns0 + i) * Qpts.col(j)
                 * Pkp1_at_Qpts.col(k);
        wcoeffs(tdim * nv + i + ns * j1, psize * j2 + k) = -w.sum();
        wcoeffs(tdim * nv + i + ns * j2, psize * j1 + k) = w.sum();
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

  // Dual space

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * tdim);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Integral representation for the boundary dofs

  // Tabulate a polynomial set on a reference edge

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (_degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(1, quad_deg);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pq_at_QptsE = PolynomialSet::tabulate_polynomial_set(Cell::Type::interval,
                                                           _degree, QptsE);
  // Iterate over edges
  for (int i = 0; i < 6; ++i)
  {
    // FIXME: get tangent from the simplex class
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> edge
        = simplex_cell.sub_entity_geometry(1, i);
    Eigen::Vector3d tangent = edge.row(1) - edge.row(0);

    // Map quadrature points onto tetrahedron edge
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        QptsE_scaled(QptsE.rows(), tdim);
    for (int j = 0; j < QptsE.rows(); ++j)
      QptsE_scaled.row(j)
          = edge.row(0) + QptsE(j, 0) * (edge.row(1) - edge.row(0));

    // Tabulate Pkp1 at edge quadrature points
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsE = PolynomialSet::tabulate_polynomial_set(
                            Cell::Type::tetrahedron, _degree + 1, QptsE_scaled)
                            .transpose();

    // Compute edge tangent integral moments
    for (int j = 0; j < Pq_at_QptsE.cols(); ++j)
    {
      Eigen::ArrayXd phi = Pq_at_QptsE.col(j);
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
    // Create quadrature scheme on the facet
    int quad_deg = 5 * (_degree + 1);
    auto [QptsF, QwtsF] = make_quadrature(2, quad_deg);
    // Tabulate a polynomial set on a reference facet
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        PqF_at_QptsF = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::triangle, _degree - 1, QptsF);

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          facet = simplex_cell.sub_entity_geometry(2, i);
      // Face tangents
      Eigen::Vector3d t0 = facet.row(1) - facet.row(0);
      Eigen::Vector3d t1 = facet.row(2) - facet.row(0);

      // Map quadrature points onto tetrahedron facet
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          QptsF_scaled(QptsF.rows(), tdim);
      for (int j = 0; j < QptsF.rows(); ++j)
        QptsF_scaled.row(j) = facet.row(0)
                              + QptsF(j, 0) * (facet.row(1) - facet.row(0))
                              + QptsF(j, 1) * (facet.row(2) - facet.row(0));

      // Tabulate Pkp1 at facet quadrature points
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          Pkp1_at_QptsF
          = PolynomialSet::tabulate_polynomial_set(Cell::Type::tetrahedron,
                                                   _degree + 1, QptsF_scaled)
                .transpose();

      for (int j = 0; j < PqF_at_QptsF.cols(); ++j)
      {
        for (int k = 0; k < tdim; ++k)
        {
          // FIXME: check this over
          Eigen::ArrayXd phi = PqF_at_QptsF.col(j);
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
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkm2_at_Qpts = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::tetrahedron, _degree - 2, Qpts);
    for (int i = 0; i < Pkm2_at_Qpts.cols(); ++i)
    {
      Eigen::ArrayXd phi = Pkm2_at_Qpts.col(i);
      Eigen::VectorXd q = phi * Qwts;
      Eigen::RowVectorXd qcoeffs = Pkp1_at_Qpts.matrix().transpose() * q;
      assert(qcoeffs.size() == psize);
      for (int j = 0; j < tdim; ++j)
      {
        dualmat.block(c, psize * j, 1, psize) = qcoeffs;
        ++c;
      }
    }
  }

  apply_dualmat_to_basis(wcoeffs, dualmat);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Nedelec2D::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_pts
      = PolynomialSet::tabulate_polynomial_set(_cell_type, _degree + 1, pts);
  const int psize = Pkp1_at_pts.cols();
  const int ndofs = _new_coeffs.rows();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), ndofs * tdim);
  result.setZero();

  for (int j = 0; j < tdim; ++j)
    for (int i = 0; i < ndofs; ++i)
      for (int k = 0; k < psize; ++k)
        result.col(i + ndofs * j)
            += Pkp1_at_pts.col(k) * _new_coeffs(i, k + psize * j);

  return result;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Nedelec3D::tabulate_basis(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_pts
      = PolynomialSet::tabulate_polynomial_set(_cell_type, _degree + 1, pts);
  const int psize = Pkp1_at_pts.cols();
  const int ndofs = _new_coeffs.rows();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), ndofs * tdim);
  result.setZero();

  for (int j = 0; j < tdim; ++j)
    for (int i = 0; i < ndofs; ++i)
      for (int k = 0; k < psize; ++k)
        result.col(i + ndofs * j)
            += Pkp1_at_pts.col(k) * _new_coeffs(i, k + psize * j);

  return result;
}
//-----------------------------------------------------------------------------
