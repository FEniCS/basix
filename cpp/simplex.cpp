// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "simplex.h"

//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceSimplex::create_simplex(int dim)
{
  if (dim < 1 or dim > 3)
    throw std::runtime_error("Unsupported dim");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      ref_geom(dim + 1, dim);
  if (dim == 1)
    ref_geom << 0.0, 1.0;
  else if (dim == 2)
    ref_geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
  else if (dim == 3)
    ref_geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  return ref_geom;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceSimplex::sub(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>& simplex,
                      int dim, int index)
{
  const int simplex_dim = simplex.rows() - 1;

  if (dim == 0)
  {
    assert(index >= 0 and index < simplex.rows());
    return simplex.row(index);
  }
  else if (dim == simplex_dim)
  {
    assert(index == 0);
    return simplex;
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> entity;

  if (dim == 1 and simplex_dim == 2)
  {
    assert(index >= 0 and index < 3);
    // Edge of triangle
    entity.resize(2, 2);
    entity.row(0) = simplex.row((index + 1) % 3);
    entity.row(1) = simplex.row((index + 2) % 3);
  }
  else if (dim == 2 and simplex_dim == 3)
  {
    assert(index >= 0 and index < 4);
    // Facet of tetrahedron
    entity.resize(3, 3);
    entity.row(0) = simplex.row((index + 1) % 4);
    entity.row(1) = simplex.row((index + 2) % 4);
    entity.row(2) = simplex.row((index + 3) % 4);
  }
  else if (dim == 1 and simplex_dim == 3)
  {
    entity.resize(2, 3);
    static const int edges[6][2]
        = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
    entity.row(0) = simplex.row(edges[index][0]);
    entity.row(1) = simplex.row(edges[index][1]);
  }
  return entity;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceSimplex::create_lattice(
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& vertices,
    int n, bool exterior)
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

  int b = (exterior == false) ? 1 : 0;

  int m = 1;
  for (int j = 0; j < tdim; ++j)
  {
    m *= (n - (tdim + 1) * b + j + 1);
    m /= (j + 1);
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points(
      m, gdim);

  int c = 0;
  if (tdim == 3)
  {
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        for (int k = b; k < n + 1 - i - j - b; ++k)
          points.row(c++)
              = vertices.row(0) + hs.row(2) * k + hs.row(1) * j + hs.row(0) * i;
  }
  else if (tdim == 2)
  {
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        points.row(c++) = vertices.row(0) + hs.row(1) * j + hs.row(0) * i;
  }
  else
  {
    for (int i = b; i < n + 1 - b; ++i)
      points.row(c++) = vertices.row(0) + hs.row(0) * i;
  }

  return points;
}
