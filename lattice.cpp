#include "lattice.h"
#include <iostream>

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_lattice(int n, const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>& vertices)
{
  int tdim = vertices.rows() - 1;
  int gdim = vertices.cols();
  assert(gdim > 0 and gdim < 4);
  assert(tdim <= gdim);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
      tdim, vertices.cols());
  for (int j = 1; j < tdim + 1; ++j)
    hs.row(tdim - j)
        = (vertices.row(j) - vertices.row(0)) / static_cast<double>(n);

  int m = 1;
  for (int j = 0; j < tdim; ++j)
  {
    m *= (n + j + 1);
    m /= (j + 1);
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points(
      m, gdim);

  if (tdim == 3)
  {
    int c = 0;
    for (int i = 0; i < n + 1; ++i)
      for (int j = 0; j < n + 1 - i; ++j)
        for (int k = 0; k < n + 1 - i - j; ++k)
          points.row(c++)
              = vertices.row(0) + hs.row(2) * k + hs.row(1) * j + hs.row(0) * i;
  }
  else if (tdim == 2)
  {
    int c = 0;
    for (int i = 0; i < n + 1; ++i)
      for (int j = 0; j < n + 1 - i; ++j)
        points.row(c++) = vertices.row(0) + hs.row(1) * j + hs.row(0) * i;
  }
  else
  {
    for (int i = 0; i < n + 1; ++i)
      points.row(i) = vertices.row(0) + hs.row(0) * i;
  }

  return points;
}
