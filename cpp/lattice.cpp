// Copyright (c) 2020 Chris Richardson & Garth Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "lattice.h"
#include "cell.h"
#include "lagrange.h"
#include "quadrature.h"
#include <Eigen/Dense>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
Eigen::ArrayXd warp_function(int n, Eigen::ArrayXd& x)
{
  [[maybe_unused]] auto [pts, wts]
      = quadrature::gauss_lobatto_legendre_line_rule(n + 1);
  wts.setZero();

  pts *= 0.5;
  for (int i = 0; i < n + 1; ++i)
    pts[i] += (0.5 - static_cast<double>(i) / static_cast<double>(n));

  FiniteElement L = create_dlagrange(cell::type::interval, n);
  Eigen::MatrixXd v = L.tabulate(0, x)[0];
  return v * pts.matrix();
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
Eigen::ArrayXXd lattice::create(cell::type celltype, int n,
                                lattice::type lattice_type, bool exterior)
{
  const double h = 1.0 / static_cast<double>(n);

  switch (celltype)
  {
  case cell::type::point:
    return Eigen::ArrayXXd::Zero(1, 1);
  case cell::type::interval:
  {
    if (n == 0)
      return Eigen::ArrayXXd::Constant(1, 1, 0.5);

    Eigen::ArrayXd x;
    if (exterior)
      x = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      x = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);

    if (lattice_type == lattice::type::gll_warped)
      x += warp_function(n, x);

    return x;
  }
  case cell::type::quadrilateral:
  {
    if (n == 0)
      return Eigen::ArrayXXd::Constant(1, 2, 0.5);

    Eigen::ArrayXd r;
    if (exterior)
      r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      r = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);

    if (lattice_type == lattice::type::gll_warped)
      r += warp_function(n, r);

    const int m = r.size();
    Eigen::ArrayX2d x(m * m, 2);
    int c = 0;
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i)
        x.row(c++) << r[i], r[j];

    return x;
  }
  case cell::type::hexahedron:
  {
    if (n == 0)
      return Eigen::ArrayXXd::Constant(1, 3, 0.5);

    Eigen::ArrayXd r;
    if (exterior)
      r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      r = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
    if (lattice_type == lattice::type::gll_warped)
      r += warp_function(n, r);

    const int m = r.size();
    Eigen::ArrayXXd x(m * m * m, 3);
    int c = 0;
    for (int k = 0; k < m; ++k)
      for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
          x.row(c++) << r[i], r[j], r[k];

    return x;
  }
  case cell::type::triangle:
  {
    if (n == 0)
      return Eigen::ArrayXXd::Constant(1, 2, 1.0 / 3.0);

    // Warp points: see Hesthaven and Warburton, Nodal Discontinuous Galerkin
    // Methods, pp. 175-180

    const int b = exterior ? 0 : 1;

    // Points
    Eigen::ArrayX2d p((n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2);

    // Displacement from GLL points in 1D, scaled by 1/(r(1-r))
    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(2 * n + 1, 0.0, 1.0);
    Eigen::ArrayXd wbar = warp_function(n, r);
    const auto s = r.segment(1, 2 * n - 1);
    wbar.segment(1, 2 * n - 1) /= s * (1 - s);

    int c = 0;
    for (int j = b; j < (n - b + 1); ++j)
    {
      for (int i = b; i < (n - b + 1 - j); ++i)
      {
        const int l = n - j - i;
        const double x = r[2 * i];
        const double y = r[2 * j];
        const double a = r[2 * l];
        p.row(c) << x, y;
        if (lattice_type == lattice::type::gll_warped)
        {
          p(c, 0) += x * (a * wbar(n + i - l) + y * wbar(n + i - j));
          p(c, 1) += y * (a * wbar(n + j - l) + x * wbar(n + j - i));
        }

        ++c;
      }
    }

    return p;
  }
  case cell::type::tetrahedron:
  {
    if (n == 0)
      return Eigen::ArrayXXd::Constant(1, 3, 0.25);

    const int b = exterior ? 0 : 1;
    Eigen::ArrayX3d p((n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6,
                      3);
    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(2 * n + 1, 0.0, 1.0);
    Eigen::ArrayXd wbar = warp_function(n, r);
    const auto s = r.segment(1, 2 * n - 1);
    wbar.segment(1, 2 * n - 1) /= s * (1 - s);
    int c = 0;
    for (int k = b; k < (n - b + 1); ++k)
    {
      for (int j = b; j < (n - b + 1 - k); ++j)
      {
        for (int i = b; i < (n - b + 1 - j - k); ++i)
        {
          const int l = n - k - j - i;
          const double x = r[2 * i];
          const double y = r[2 * j];
          const double z = r[2 * k];
          const double a = r[2 * l];
          p.row(c) << x, y, z;
          if (lattice_type == lattice::type::gll_warped)
          {
            const double dx = x
                              * (a * wbar(n + i - l) + y * wbar(n + i - j)
                                 + z * wbar(n + i - k));
            const double dy = y
                              * (a * wbar(n + j - l) + z * wbar(n + j - k)
                                 + x * wbar(n + j - i));
            const double dz = z
                              * (a * wbar(n + k - l) + x * wbar(n + k - i)
                                 + y * wbar(n + k - j));
            p(c, 0) += dx;
            p(c, 1) += dy;
            p(c, 2) += dz;
          }

          ++c;
        }
      }
    }

    return p;
  }
  case cell::type::prism:
  {
    if (n == 0)
    {
      Eigen::ArrayXXd x = Eigen::ArrayXXd::Constant(1, 3, 1.0 / 3.0);
      x(0, 2) = 0.5;
      return x;
    }

    const Eigen::ArrayXXd tri_pts
        = lattice::create(cell::type::triangle, n, lattice_type, exterior);
    const Eigen::ArrayXXd line_pts
        = lattice::create(cell::type::interval, n, lattice_type, exterior);

    Eigen::ArrayX3d x(tri_pts.rows() * line_pts.rows(), 3);
    x.leftCols(2) = tri_pts.replicate(line_pts.rows(), 1);
    for (int i = 0; i < line_pts.rows(); ++i)
      x.block(i * tri_pts.rows(), 2, tri_pts.rows(), 1) = line_pts(i, 0);
    return x;
  }
  case cell::type::pyramid:
  {
    if (n == 0)
    {
      Eigen::ArrayXXd x = Eigen::ArrayXXd::Constant(1, 3, 0.4);
      x(0, 2) = 0.2;
      return x;
    }
    else
    {
      // Interpolate warp factor along interval
      std::tuple<Eigen::ArrayXXd, Eigen::ArrayXd> pw
          = quadrature::gauss_lobatto_legendre_line_rule(n + 1);
      Eigen::VectorXd pts = std::get<0>(pw) * 0.5;
      for (int i = 0; i < n + 1; ++i)
        pts[i] += (0.5 - static_cast<double>(i) / static_cast<double>(n));
      FiniteElement L = create_dlagrange(cell::type::interval, n);

      // Get interpolated value at r in range [-1, 1]
      auto w = [&](double r) {
        Eigen::ArrayXd rr = Eigen::ArrayXd::Constant(1, 0.5 * (r + 1.0));
        Eigen::VectorXd v = L.tabulate(0, rr)[0].row(0);
        return v.dot(pts);
      };

      int b = (exterior == false) ? 1 : 0;
      n -= b * 3;
      int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
      Eigen::ArrayX3d points(m, 3);
      int c = 0;
      for (int k = 0; k < n + 1; ++k)
        for (int j = 0; j < n + 1 - k; ++j)
          for (int i = 0; i < n + 1 - k; ++i)
          {
            double x = h * (i + b);
            double y = h * (j + b);
            double z = h * (k + b);

            if (lattice_type == lattice::type::gll_warped)
            {
              // Barycentric coordinates of triangle in x-z plane
              const double l1 = x;
              const double l2 = z;
              const double l3 = 1 - x - z;
              // Barycentric coordinates of triangle in y-z plane
              const double l4 = y;
              const double l5 = z;
              const double l6 = 1 - y - z;

              // b1-b6 are the blending factors for each edge
              double b1, f1, f2;
              if (std::fabs(l1) < 1e-12)
              {
                b1 = 1.0;
                f1 = 0.0;
                f2 = 0.0;
              }
              else
              {
                b1 = 2.0 * l3 / (2.0 * l3 + l1) * 2.0 * l2 / (2.0 * l2 + l1);
                f1 = l1 / (l1 + l4);
                f2 = l1 / (l1 + l6);
              }

              // r1-r4 are the edge positions for each of the z>0 edges
              // calculated so that they use the barycentric coordinates
              // of the triangle, if the point lies on a triangular face.
              // f1-f4 are face selecting functions, which blend between
              // adjacent triangular faces
              const double r1 = (l2 - l3) * f1 + (l5 - l6) * (1 - f1);
              const double r2 = (l2 - l3) * f2 + (l5 - l4) * (1 - f2);

              double b2;
              if (std::fabs(l2) < 1e-12)
                b2 = 1.0;
              else
                b2 = 2.0 * l3 / (2.0 * l3 + l2) * 2.0 * l1 / (2.0 * l1 + l2);

              double b3, f3, f4;
              if (std::fabs(l3) < 1e-12)
              {
                b3 = 1.0;
                f3 = 0.0;
                f4 = 0.0;
              }
              else
              {
                b3 = 2.0 * l2 / (2.0 * l2 + l3) * 2.0 * l1 / (2.0 * l1 + l3);
                f3 = l3 / (l3 + l4);
                f4 = l3 / (l3 + l6);
              }

              const double r3 = (l2 - l1) * f3 + (l5 - l6) * (1.0 - f3);
              const double r4 = (l2 - l1) * f4 + (l5 - l4) * (1.0 - f4);

              double b4;
              if (std::fabs(l4) < 1e-12)
                b4 = 1.0;
              else
                b4 = 2 * l6 / (2.0 * l6 + l4) * 2.0 * l5 / (2.0 * l5 + l4);

              double b5;
              if (std::fabs(l5) < 1e-12)
                b5 = 1.0;
              else
                b5 = 2.0 * l6 / (2.0 * l6 + l5) * 2.0 * l4 / (2.0 * l4 + l5);

              double b6;
              if (std::fabs(l6) < 1e-12)
                b6 = 1.0;
              else
                b6 = 2.0 * l4 / (2.0 * l4 + l6) * 2.0 * l5 / (2.0 * l5 + l6);

              double dx = -b3 * b4 * w(r3) - b3 * b6 * w(r4) + b2 * w(l1 - l3);
              double dy = -b1 * b6 * w(r2) - b3 * b6 * w(r4) + b5 * w(l4 - l6);
              double dz = b1 * b4 * w(r1) + b1 * b6 * w(r2) + b3 * b4 * w(r3)
                          + b3 * b6 * w(r4);

              x += dx;
              y += dy;
              z += dz;
            }

            points.row(c++) << x, y, z;
          }

      return points;
    }
  }
  default:
    throw std::runtime_error("Unsupported cell for lattice");
  }
}

//-----------------------------------------------------------------------------
