#include "lattice.h"
#include "cell.h"
#include "lagrange.h"
#include "quadrature.h"
#include <Eigen/Dense>

using namespace libtab;

namespace
{
Eigen::ArrayXd warp_function(int n, Eigen::ArrayXd& x)
{
  [[maybe_unused]] auto [pts, wts]
      = quadrature::gauss_lobatto_legendre_line_rule(n + 1);
  wts.setZero();

  pts *= 0.5;
  for (int i = 0; i < n + 1; ++i)
    pts[i] += (0.5 - static_cast<double>(i) / static_cast<double>(n));

  FiniteElement L = DiscontinuousLagrange::create(cell::Type::interval, n);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v
      = L.tabulate(0, x)[0];
  Eigen::ArrayXd w(v.rows());
  w = v * pts.matrix();

  return w;
}

} // namespace

//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
lattice::create(cell::Type celltype, int n, lattice::Type lattice_type,
                bool exterior)
{
  const double h = 1.0 / static_cast<double>(n);

  if (celltype == cell::Type::point)
  {
    Eigen::ArrayXd x = Eigen::ArrayXd::Zero(1, 1);
    return x;
  }
  if (celltype == cell::Type::interval)
  {
    if (n == 0)
    {
      Eigen::ArrayXd x(1, 1);
      x(0, 0) = 0.5;
      return x;
    }

    Eigen::ArrayXd x;
    if (exterior)
      x = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      x = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
    if (lattice_type == lattice::Type::gll_warped)
      x += warp_function(n, x);
    return x;
  }
  else if (celltype == cell::Type::quadrilateral)
  {
    if (n == 0)
    {
      Eigen::ArrayXd x(1, 2);
      x.row(0) << 0.5, 0.5;
      return x;
    }

    Eigen::ArrayXd r;
    if (exterior)
      r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      r = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
    if (lattice_type == lattice::Type::gll_warped)
      r += warp_function(n, r);

    const int m = r.size();
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        m * m, 2);

    int c = 0;
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i)
        x.row(c++) << r[i], r[j];

    return x;
  }
  else if (celltype == cell::Type::hexahedron)
  {
    if (n == 0)
    {
      Eigen::ArrayXd x(1, 3);
      x.row(0) << 0.5, 0.5, 0.5;
      return x;
    }

    Eigen::ArrayXd r;
    if (exterior)
      r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      r = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
    if (lattice_type == lattice::Type::gll_warped)
      r += warp_function(n, r);

    const int m = r.size();
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        m * m * m, 3);

    int c = 0;
    for (int k = 0; k < m; ++k)
      for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
          x.row(c++) << r[i], r[j], r[k];

    return x;
  }
  else if (celltype == cell::Type::triangle)
  {
    if (n == 0)
    {
      Eigen::ArrayXd x(1, 2);
      x.row(0) << 1.0 / 3.0, 1.0 / 3.0;
      return x;
    }

    // Warp points: see Hesthaven and Warburton, Nodal Discontinuous Galerkin
    // Methods, pp. 175-180

    const int b = exterior ? 0 : 1;

    // Points
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
        (n - 3 * b + 1) * (n - 3 * b + 2) / 2, 2);

    // Displacement from GLL points in 1D, scaled by 1/(r(1-r))
    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(2 * n + 1, 0.0, 1.0);
    Eigen::ArrayXd wbar = warp_function(n, r);
    const auto s = r.segment(1, 2 * n - 1);
    wbar.segment(1, 2 * n - 1) /= (s * (1 - s));

    int c = 0;
    for (int j = b; j < (n - b + 1); ++j)
      for (int i = b; i < (n - b + 1 - j); ++i)
      {
        int l = n - j - i;
        const double x = r[2 * i];
        const double y = r[2 * j];
        const double a = r[2 * l];

        p.row(c) << x, y;
        if (lattice_type == lattice::Type::gll_warped)
        {
          const double dx = x * (a * wbar(n + i - l) + y * wbar(n + i - j));
          const double dy = y * (a * wbar(n + j - l) + x * wbar(n + j - i));
          p(c, 0) += dx;
          p(c, 1) += dy;
        }

        ++c;
      }
    return p;
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    if (n == 0)
    {
      Eigen::ArrayXd x(1, 3);
      x.row(0) << 0.25, 0.25, 0.25;
      return x;
    }

    const int b = exterior ? 0 : 1;

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
        (n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6, 3);

    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(2 * n + 1, 0.0, 1.0);
    Eigen::ArrayXd wbar = warp_function(n, r);
    const auto s = r.segment(1, 2 * n - 1);
    wbar.segment(1, 2 * n - 1) /= (s * (1 - s));

    int c = 0;
    for (int k = b; k < (n - b + 1); ++k)
      for (int j = b; j < (n - b + 1 - k); ++j)
        for (int i = b; i < (n - b + 1 - j - k); ++i)
        {
          int l = n - k - j - i;
          const double x = r[2 * i];
          const double y = r[2 * j];
          const double z = r[2 * k];
          const double a = r[2 * l];

          p.row(c) << x, y, z;
          if (lattice_type == lattice::Type::gll_warped)
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
    return p;
  }
  else if (celltype == cell::Type::prism)
  {
    if (n == 0)
    {
      Eigen::ArrayXd x(1, 3);
      x.row(0) << 1.0 / 3.0, 1.0 / 3.0, 0.5;
      return x;
    }
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        tri_pts
        = lattice::create(cell::Type::triangle, n, lattice_type, exterior);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        line_pts
        = lattice::create(cell::Type::interval, n, lattice_type, exterior);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        tri_pts.rows() * line_pts.rows(), 3);

    x.leftCols(2) = tri_pts.replicate(line_pts.rows(), 1);
    for (int i = 0; i < line_pts.rows(); ++i)
      x.block(i * tri_pts.rows(), 2, tri_pts.rows(), 1) = line_pts(i, 0);
    return x;
  }
  else if (celltype == cell::Type::pyramid)
  {
    if (lattice_type == lattice::Type::gll_warped)
      throw std::runtime_error(
          "Warped lattice not yet implemented for pyramid");

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        points;

    if (n == 0)
    {
      points.resize(1, 3);
      points.row(0) << 0.4, 0.4, 0.2;
    }
    else
    {
      int b = (exterior == false) ? 1 : 0;
      n -= b * 3;

      int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
      points.resize(m, 3);
      int c = 0;
      for (int k = 0; k < n + 1; ++k)
        for (int j = 0; j < n + 1 - k; ++j)
          for (int i = 0; i < n + 1 - k; ++i)
            points.row(c++) << h * (i + b), h * (j + b), h * (k + b);
    }
    return points;
  }
  else
    throw std::runtime_error("Unsupported cell for lattice");
}
