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

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
lattice::create(cell::Type celltype, int n, bool exterior)
{
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points;

  const double h = 1.0 / static_cast<double>(n);

  if (celltype == cell::Type::quadrilateral)
  {
    if (n == 0)
    {
      points.resize(1, 2);
      points.row(0) << 0.5, 0.5;
    }
    else
    {
      int b = (exterior == false) ? 1 : 0;
      int m = (n + 1 - 2 * b);
      points.resize(m * m, 2);
      int c = 0;
      for (int i = b; i < n + 1 - b; ++i)
        for (int j = b; j < n + 1 - b; ++j)
          points.row(c++) << j * h, i * h;
    }
  }
  else if (celltype == cell::Type::hexahedron)
  {
    if (n == 0)
    {
      points.resize(1, 3);
      points.row(0) << 0.5, 0.5, 0.5;
    }
    else
    {
      int b = (exterior == false) ? 1 : 0;
      int m = (n - 2 * b + 1);
      points.resize(m * m * m, 3);
      int c = 0;
      for (int i = b; i < n + 1 - b; ++i)
        for (int j = b; j < n + 1 - b; ++j)
          for (int k = b; k < n + 1 - b; ++k)
            points.row(c++) << k * h, j * h, i * h;
    }
  }
  else if (celltype == cell::Type::point)
  {
    points.resize(1, 1);
    points(0, 0) = 0.0;
  }
  else if (celltype == cell::Type::interval)
  {
    if (n == 0)
    {
      points.resize(1, 1);
      points(0, 0) = 0.5;
    }
    else
    {
      if (exterior)
      {
        points.resize(n + 1, 1);
        points.col(0) = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
      }
      else
      {
        points.resize(n - 1, 1);
        points.col(0) = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
      }
    }
  }
  else if (celltype == cell::Type::triangle)
  {
    if (n == 0)
    {
      points.resize(1, 2);
      points.row(0) << 1.0 / 3, 1.0 / 3;
    }
    else
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
          2, 2);
      hs.row(0) << 0.0, 1.0;
      hs.row(1) << 1.0, 0.0;
      hs /= static_cast<double>(n);

      int b = (exterior == false) ? 1 : 0;
      int m = (n - 3 * b + 1) * (n - 3 * b + 2) / 2;
      points.resize(m, 2);
      int c = 0;
      for (int i = b; i < n + 1 - b; ++i)
        for (int j = b; j < n + 1 - i - b; ++j)
          points.row(c++) = hs.row(1) * j + hs.row(0) * i;
    }
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    if (n == 0)
    {
      points.resize(1, 3);
      points.row(0) << 1.0 / 4, 1.0 / 4, 1.0 / 4;
    }
    else
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
          3, 3);
      hs.row(0) << 0.0, 0.0, 1.0;
      hs.row(1) << 0.0, 1.0, 0.0;
      hs.row(2) << 1.0, 0.0, 0.0;
      hs /= static_cast<double>(n);

      int b = (exterior == false) ? 1 : 0;
      int m = (n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6;
      points.resize(m, 3);
      int c = 0;
      for (int i = b; i < n + 1 - b; ++i)
        for (int j = b; j < n + 1 - i - b; ++j)
          for (int k = b; k < n + 1 - i - j - b; ++k)
            points.row(c++) = hs.row(2) * k + hs.row(1) * j + hs.row(0) * i;
    }
  }
  else if (celltype == cell::Type::prism)
  {
    if (n == 0)
    {
      points.resize(1, 3);
      points.row(0) << 1.0 / 3, 1.0 / 3, 0.5;
    }
    else
    {

      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
          3, 3);

      hs.row(0) << 0.0, 0.0, 1.0;
      hs.row(1) << 0.0, 1.0, 0.0;
      hs.row(2) << 1.0, 0.0, 0.0;
      hs /= static_cast<double>(n);

      int b = (exterior == false) ? 1 : 0;
      int m = (n - 2 * b + 1) * (n - 3 * b + 1) * (n - 3 * b + 2) / 2;
      points.resize(m, 3);
      int c = 0;
      for (int i = b; i < n + 1 - b; ++i)
        for (int j = b; j < n + 1 - i - b; ++j)
          for (int k = b; k < n + 1 - b; ++k)
            points.row(c++) = hs.row(0) * k + hs.row(1) * j + hs.row(2) * i;
    }
  }
  else if (celltype == cell::Type::pyramid)
  {
    if (n == 0)
    {
      points.resize(1, 3);
      points.row(0) << 0.5, 0.5, 1.0 / 5;
    }
    else
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
          3, 3);

      hs.row(0) << 0.0, 0.0, 1.0;
      hs.row(1) << 0.0, 1.0, 0.0;
      hs.row(2) << 1.0, 0.0, 0.0;
      hs /= static_cast<double>(n);

      int b = (exterior == false) ? 1 : 0;
      n -= b * 3;

      int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
      points.resize(m, 3);
      int c = 0;
      for (int k = 0; k < n + 1; ++k)
        for (int i = 0; i < n + 1 - k; ++i)
          for (int j = 0; j < n + 1 - k; ++j)
            points.row(c++) = hs.row(0) * (k + b) + hs.row(1) * (j + b)
                              + hs.row(2) * (i + b);
    }
  }
  else
    throw std::runtime_error("Unsupported cell for lattice");

  return points;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
lattice::create_warped(cell::Type celltype, int n, bool exterior)
{
  const double h = 1.0 / static_cast<double>(n);

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
    x += warp_function(n, x);
    return x;
  }
  else if (celltype == cell::Type::quadrilateral)
  {
    Eigen::ArrayXd r;
    if (exterior)
      r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      r = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
    r += warp_function(n, r);

    const int m = r.size();
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        m * m, 2);

    int c = 0;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
      {
        x(c, 0) = r[i];
        x(c, 1) = r[j];
        ++c;
      }
    return x;
  }
  else if (celltype == cell::Type::hexahedron)
  {
    Eigen::ArrayXd r;
    if (exterior)
      r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    else
      r = Eigen::VectorXd::LinSpaced(n - 1, h, 1.0 - h);
    r += warp_function(n, r);

    const int m = r.size();
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        m * m * m, 3);
    int c = 0;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < m; ++k)
        {
          x(c, 0) = r[i];
          x(c, 1) = r[j];
          x(c, 2) = r[k];
          ++c;
        }
    return x;
  }
  else if (celltype == cell::Type::triangle)
  {
    // Warp points: see Hesthaven and Warburton, Nodal Discontinuous Galerkin
    // Methods, pp. 175-180

    // Points
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
        (n + 2) * (n + 1) / 2, 2);

    // Displacement from GLL points in 1D, scaled by 1/(r(1-r))
    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(2 * n + 1, 0.0, 1.0);
    Eigen::ArrayXd wbar = warp_function(n, r);
    r[0] = 0.5;
    r[2 * n] = 0.5;
    wbar /= (r * (1 - r));

    int c = 0;
    for (int i = 0; i < n + 1; ++i)
      for (int j = 0; j < (n + 1 - i); ++j)
      {
        int l = n - j - i;
        const double x = r[2 * i];
        const double y = r[2 * j];
        const double a = r[2 * l];

        const double dx = x * (a * wbar(n + i - l) + y * wbar(n + i - j));
        const double dy = y * (a * wbar(n + j - l) + x * wbar(n + j - i));

        p(c, 0) = x + dx;
        p(c, 1) = y + dy;
        ++c;
      }
    return p;
  }
  else if (celltype == cell::Type::tetrahedron)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
        (n + 3) * (n + 2) * (n + 1) / 6, 3);

    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(2 * n + 1, 0.0, 1.0);
    Eigen::ArrayXd wbar = warp_function(n, r);
    r[0] = 0.5;
    r[2 * n] = 0.5;
    wbar /= (r * (1 - r));

    int c = 0;
    for (int i = 0; i < n + 1; ++i)
      for (int j = 0; j < (n + 1 - i); ++j)
        for (int k = 0; k < (n + 1 - j - i); ++k)
        {
          int l = n - k - j - i;
          const double x = i * h;
          const double y = j * h;
          const double z = k * h;
          const double a = l * h;

          const double dx = x
                            * (a * wbar(n + i - l) + y * wbar(n + i - j)
                               + z * wbar(n + i - k));
          const double dy = y
                            * (a * wbar(n + j - l) + z * wbar(n + j - k)
                               + x * wbar(n + j - i));
          const double dz = z
                            * (a * wbar(n + k - l) + x * wbar(n + k - i)
                               + y * wbar(n + k - j));

          p(c, 0) = x + dx;
          p(c, 1) = y + dy;
          p(c, 2) = z + dz;
          ++c;
        }
    return p;
  }
  else
    throw std::runtime_error("Unsupported cell type");
}
