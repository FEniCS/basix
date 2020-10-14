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

namespace
{

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_space(int degree)
{
  // 2D space on triangle
  const int tdim = 2;

  // Vector subset
  const int nv = (degree + 1) * (degree + 2) / 2;

  // PkH subset
  const int ns = degree + 1;
  const int ns0 = (degree + 1) * degree / 2;

  // Tabulate P(k+1) at quadrature points
  auto [Qpts, Qwts] = make_quadrature(tdim, 2 * degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
          Cell::Type::triangle, degree + 1, Qpts);

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
  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_2d_dual(int degree)
{
  // 2D triangle
  const int tdim = 2;

  // Tabulate P(k+1) at quadrature points
  const int ndofs = 3 * (degree + 1) + degree * (degree + 1);
  auto [Qpts, Qwts] = make_quadrature(tdim, 2 * degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
          Cell::Type::triangle, degree + 1, Qpts);
  const int psize = Pkp1_at_Qpts.cols();

  // Dual space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * 2);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Integral representation for the boundary (edge) dofs

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(1, quad_deg);

  // Tabulate a polynomial set on a reference edge
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pq_at_QptsE = PolynomialSet::tabulate_polynomial_set(Cell::Type::interval,
                                                           degree, QptsE);
  // Iterate over edges
  for (int i = 0; i < 3; ++i)
  {
    // FIXME: get edge tangent from the cell class
    Eigen::Array<double, 2, 2, Eigen::RowMajor> edge
        = Cell::sub_entity_geometry(Cell::Type::triangle, 1, i);
    Eigen::Vector2d tangent = edge.row(1) - edge.row(0);

    // Map quadrature points onto triangle edge
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        QptsE_scaled(QptsE.rows(), 2);
    for (int j = 0; j < QptsE.rows(); ++j)
      QptsE_scaled.row(j)
          = edge.row(0) + QptsE(j, 0) * (edge.row(1) - edge.row(0));

    // Tabulate Pkp1 at edge quadrature points
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsE = PolynomialSet::tabulate_polynomial_set(
                            Cell::Type::triangle, degree + 1, QptsE_scaled)
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

  if (degree > 0)
  {
    // Interior integral moment
    auto [QptsI, QwtsI] = make_quadrature(2, quad_deg);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkm1_at_QptsI = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::triangle, degree - 1, QptsI);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsI = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::triangle, degree + 1, QptsI);

    for (int i = 0; i < Pkm1_at_QptsI.cols(); ++i)
    {
      Eigen::ArrayXd phi = Pkm1_at_QptsI.col(i);
      Eigen::VectorXd q = phi * QwtsI;
      Eigen::RowVectorXd qcoeffs = Pkp1_at_QptsI.matrix().transpose() * q;
      assert(qcoeffs.size() == psize);
      dualmat.block(c, 0, 1, psize) = qcoeffs;
      ++c;
      dualmat.block(c, psize, 1, psize) = qcoeffs;
      ++c;
    }
  }
  return dualmat;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const int tdim = 3;

  // Vector subset
  const int nv = (degree + 1) * (degree + 2) * (degree + 3) / 6;

  // PkH subset
  const int ns = (degree + 1) * (degree + 2) / 2;
  const int ns0 = degree * (degree + 1) * (degree + 2) / 6;

  // Tabulate P(k+1) at quadrature points
  auto [Qpts, Qwts] = make_quadrature(tdim, 2 * degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
          Cell::Type::tetrahedron, degree + 1, Qpts);
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

  int ndofs = 6 * (degree + 1) + 4 * degree * (degree + 1)
              + (degree - 1) * degree * (degree + 1) / 2;
  wcoeffs = svd.matrixV().transpose().topRows(ndofs);

  // Check singular values (should only be ndofs which are significant)
  Eigen::VectorXd s = svd.singularValues();
  for (int i = 0; i < ndofs; ++i)
    if (s[i] < 1e-12)
      throw std::runtime_error("Error in Nedelec3D space");
  for (int i = ndofs; i < s.size(); ++i)
    if (s[i] > 1e-12)
      throw std::runtime_error("Error in Nedelec3D space");

  return wcoeffs;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_nedelec_3d_dual(int degree)
{
  const int tdim = 3;

  // Tabulate P(k+1) at quadrature points
  auto [Qpts, Qwts] = make_quadrature(tdim, 2 * degree + 2);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pkp1_at_Qpts = PolynomialSet::tabulate_polynomial_set(
          Cell::Type::tetrahedron, degree + 1, Qpts);
  const int psize = Pkp1_at_Qpts.cols();

  // Work out number of dofs
  const int ndofs = 6 * (degree + 1) + 4 * degree * (degree + 1)
                    + (degree - 1) * degree * (degree + 1) / 2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dualmat(ndofs, psize * tdim);
  dualmat.setZero();

  // dof counter
  int c = 0;

  // Integral representation for the boundary dofs

  // Tabulate a polynomial set on a reference edge

  // Create quadrature scheme on the edge
  int quad_deg = 5 * (degree + 1);
  auto [QptsE, QwtsE] = make_quadrature(1, quad_deg);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Pq_at_QptsE = PolynomialSet::tabulate_polynomial_set(Cell::Type::interval,
                                                           degree, QptsE);
  // Iterate over edges
  for (int i = 0; i < 6; ++i)
  {
    // FIXME: get tangent from the simplex class
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> edge
        = Cell::sub_entity_geometry(Cell::Type::tetrahedron, 1, i);
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
                            Cell::Type::tetrahedron, degree + 1, QptsE_scaled)
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

  if (degree > 0)
  {
    // Create quadrature scheme on the facet
    int quad_deg = 5 * (degree + 1);
    auto [QptsF, QwtsF] = make_quadrature(2, quad_deg);
    // Tabulate a polynomial set on a reference facet
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        PqF_at_QptsF = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::triangle, degree - 1, QptsF);

    for (int i = 0; i < 4; ++i)
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          facet = Cell::sub_entity_geometry(Cell::Type::tetrahedron, 2, i);
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
          Pkp1_at_QptsF = PolynomialSet::tabulate_polynomial_set(
                              Cell::Type::tetrahedron, degree + 1, QptsF_scaled)
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
  if (degree > 1)
  {
    // Interior integral moment
    auto [QptsI, QwtsI] = make_quadrature(tdim, quad_deg);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkm2_at_QptsI = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::tetrahedron, degree - 2, QptsI);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Pkp1_at_QptsI = PolynomialSet::tabulate_polynomial_set(
            Cell::Type::tetrahedron, degree + 1, QptsI);
    for (int i = 0; i < Pkm2_at_QptsI.cols(); ++i)
    {
      Eigen::ArrayXd phi = Pkm2_at_QptsI.col(i);
      Eigen::VectorXd q = phi * QwtsI;
      Eigen::RowVectorXd qcoeffs = Pkp1_at_QptsI.matrix().transpose() * q;
      assert(qcoeffs.size() == psize);
      for (int j = 0; j < tdim; ++j)
      {
        dualmat.block(c, psize * j, 1, psize) = qcoeffs;
        ++c;
      }
    }
  }
  return dualmat;
}

} // namespace

//-----------------------------------------------------------------------------
Nedelec::Nedelec(Cell::Type celltype, int k) : FiniteElement(celltype, k - 1)
{

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> wcoeffs;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dualmat;

  if (celltype == Cell::Type::triangle)
  {
    wcoeffs = create_nedelec_2d_space(_degree);
    dualmat = create_nedelec_2d_dual(_degree);
  }
  else if (celltype == Cell::Type::tetrahedron)
  {
    wcoeffs = create_nedelec_3d_space(_degree);
    dualmat = create_nedelec_3d_dual(_degree);
  }
  else
    throw std::runtime_error("Invalid celltype in Nedelec");

  apply_dualmat_to_basis(wcoeffs, dualmat);
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
Nedelec::tabulate(int nderiv,
                  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>& pts) const
{
  const int tdim = Cell::topological_dimension(_cell_type);
  if (pts.cols() != tdim)
    throw std::runtime_error(
        "Point dimension does not match element dimension");

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      Pkp1_at_pts = PolynomialSet::tabulate_polynomial_set_deriv(
          _cell_type, _degree + 1, nderiv, pts);
  const int psize = Pkp1_at_pts[0].cols();
  const int ndofs = _coeffs.rows();

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(Pkp1_at_pts.size());

  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    dresult[p].resize(pts.rows(), ndofs * tdim);
    for (int j = 0; j < tdim; ++j)
      dresult[p].block(0, ndofs * j, pts.rows(), ndofs)
          = Pkp1_at_pts[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
  }

  return dresult;
}
//-----------------------------------------------------------------------------
