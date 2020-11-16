// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "lagrange.h"
#include "quadrature.h"
#include <iostream>
#include <map>

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
cell::geometry(cell::Type celltype)
{
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom;

  switch (celltype)
  {
  case cell::Type::interval:
    geom.resize(2, 1);
    geom << 0.0, 1.0;
    break;
  case cell::Type::triangle:
    geom.resize(3, 2);
    geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
    break;
  case cell::Type::quadrilateral:
    geom.resize(4, 2);
    geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    break;
  case cell::Type::tetrahedron:
    geom.resize(4, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    break;
  case cell::Type::prism:
    geom.resize(6, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 1.0;
    break;
  case cell::Type::pyramid:
    geom.resize(5, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        0.0, 1.0;
    break;
  case cell::Type::hexahedron:
    geom.resize(8, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
  return geom;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::vector<int>>> cell::topology(cell::Type celltype)
{
  std::vector<std::vector<std::vector<int>>> topo;

  switch (celltype)
  {
  case cell::Type::interval:
    topo.resize(2);
    // Vertices
    topo[0] = {{0}, {1}};
    // Cell
    topo[1] = {{0, 1}};
    break;
  case cell::Type::triangle:
    topo.resize(3);
    // Vertices
    topo[0] = {{0}, {1}, {2}};
    // Edges
    topo[1] = {{1, 2}, {0, 2}, {0, 1}};
    // Cell
    topo[2] = {{0, 1, 2}};
    break;
  case cell::Type::quadrilateral:
    topo.resize(3);
    // FIXME - check all these
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}};
    // Edges
    topo[1] = {{0, 2}, {2, 3}, {3, 1}, {1, 0}};
    // Cell
    topo[2] = {{0, 1, 2, 3}};
    break;
  case cell::Type::tetrahedron:
    topo.resize(4);
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}};
    // Edges
    topo[1] = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
    // Faces
    topo[2] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
    // Cell
    topo[3] = {{0, 1, 2, 3}};
    break;
  case cell::Type::prism:
    // FIXME: check
    topo.resize(4);
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}, {4}, {5}};
    // Edges
    topo[1] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 4},
               {2, 5}, {3, 4}, {4, 5}, {5, 3}};
    // Faces
    topo[2] = {{0, 1, 2}, {0, 1, 3, 4}, {1, 2, 4, 5}, {2, 0, 5, 3}, {3, 4, 5}};
    // Cell
    topo[3] = {{0, 1, 2, 3, 4, 5}};
    break;
  case cell::Type::pyramid:
    // FIXME: check all these
    topo.resize(4);
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}, {4}};
    // Edges
    topo[1] = {{0, 1}, {0, 2}, {2, 3}, {3, 1}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};
    // Faces
    topo[2] = {{0, 1, 2, 3}, {0, 1, 4}, {0, 2, 4}, {2, 3, 4}, {3, 1, 4}};
    // Cell
    topo[3] = {{0, 1, 2, 3, 4}};
    break;
  case cell::Type::hexahedron:
    topo.resize(4);
    // FIXME: check over
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
    // Edges
    topo[1] = {{0, 1}, {0, 2}, {2, 3}, {3, 1}, {0, 4}, {1, 5},
               {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {7, 6}};
    // Faces
    topo[2] = {{0, 1, 2, 3}, {0, 1, 4, 5}, {1, 3, 5, 7},
               {2, 3, 6, 7}, {2, 0, 6, 4}, {4, 5, 6, 7}};
    // Cell
    topo[3] = {{0, 1, 2, 3, 4, 5, 6, 7}};
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
  return topo;
}
//-----------------------------------------------------------------------------
int cell::topological_dimension(cell::Type cell_type)
{
  switch (cell_type)
  {
  case cell::Type::interval:
    return 1;
  case cell::Type::triangle:
    return 2;
  case cell::Type::quadrilateral:
    return 2;
  case cell::Type::tetrahedron:
    return 3;
  case cell::Type::hexahedron:
    return 3;
  case cell::Type::prism:
    return 3;
  case cell::Type::pyramid:
    return 3;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
  return 0;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
cell::sub_entity_geometry(cell::Type celltype, int dim, int index)
{
  std::vector<std::vector<std::vector<int>>> cell_topology
      = cell::topology(celltype);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cell_geometry = cell::geometry(celltype);

  if (dim < 0 or dim >= (int)cell_topology.size())
    throw std::runtime_error("Invalid dimension for sub-entity");

  const std::vector<std::vector<int>>& t = cell_topology[dim];

  if (index < 0 or index >= (int)t.size())
    throw std::runtime_error("Invalid entity index");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      sub_entity(t[index].size(), cell_geometry.cols());

  for (int i = 0; i < sub_entity.rows(); ++i)
    sub_entity.row(i) = cell_geometry.row(t[index][i]);

  return sub_entity;
}
//----------------------------------------------------------------------------
int cell::sub_entity_count(cell::Type celltype, int dim)
{
  std::vector<std::vector<std::vector<int>>> cell_topology
      = cell::topology(celltype);

  if (dim < 0 or dim >= (int)cell_topology.size())
    throw std::runtime_error("Invalid dimension for sub-entity");

  const std::vector<std::vector<int>>& t = cell_topology[dim];
  return (int)t.size();
}
//----------------------------------------------------------------------------
cell::Type cell::sub_entity_type(cell::Type celltype, int dim, int index)
{
  const int tdim = cell::topological_dimension(celltype);
  assert(dim >= 0 and dim <= tdim);

  if (dim == 0)
    return cell::Type::point;
  else if (dim == 1)
    return cell::Type::interval;
  else if (dim == tdim)
    return celltype;

  const std::vector<std::vector<std::vector<int>>> t = cell::topology(celltype);
  const std::vector<int>& entity = t[dim][index];
  switch (entity.size())
  {
  case 3:
    return cell::Type::triangle;
  case 4:
    return cell::Type::quadrilateral;
  default:
    throw std::runtime_error("Error in sub_entity_type");
  }
}
//-----------------------------------------------------------------------------
cell::Type cell::str_to_type(std::string name)
{
  static const std::map<std::string, cell::Type> name_to_type
      = {{"point", cell::Type::point},
         {"interval", cell::Type::interval},
         {"triangle", cell::Type::triangle},
         {"tetrahedron", cell::Type::tetrahedron},
         {"quadrilateral", cell::Type::quadrilateral},
         {"pyramid", cell::Type::pyramid},
         {"prism", cell::Type::prism},
         {"hexahedron", cell::Type::hexahedron}};

  auto it = name_to_type.find(name);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find name " + name);

  return it->second;
}
//-----------------------------------------------------------------------------
double cell::warp(int n, double x)
{
  Eigen::ArrayXd r(1);
  r(0) = x;
  return warp_function(n, r)[0];
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
cell::warped_lattice(cell::Type celltype, int n)
{
  if (n < 1)
    throw std::runtime_error("Cannot create warped lattice for n < 1");

  if (celltype == cell::Type::interval)
  {
    Eigen::ArrayXd x = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    x += warp_function(n, x);
    return x;
  }
  else if (celltype == cell::Type::quadrilateral)
  {
    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    r += warp_function(n, r);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        (n + 1) * (n + 1), 2);
    int c = 0;
    for (int i = 0; i < n + 1; ++i)
      for (int j = 0; j < n + 1; ++j)
      {
        x(c, 0) = r[i];
        x(c, 1) = r[j];
        ++c;
      }
    return x;
  }
  else if (celltype == cell::Type::hexahedron)
  {
    Eigen::ArrayXd r = Eigen::VectorXd::LinSpaced(n + 1, 0.0, 1.0);
    r += warp_function(n, r);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
        (n + 1) * (n + 1) * (n + 1), 3);
    int c = 0;
    for (int i = 0; i < n + 1; ++i)
      for (int j = 0; j < n + 1; ++j)
        for (int k = 0; k < n + 1; ++k)
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
        double x = static_cast<double>(i) / static_cast<double>(n);
        double y = static_cast<double>(j) / static_cast<double>(n);
        double a = static_cast<double>(l) / static_cast<double>(n);

        double dx = x * (a * wbar(n + i - l) + y * wbar(n + i - j));
        double dy = y * (a * wbar(n + j - l) + x * wbar(n + j - i));

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
          double x = static_cast<double>(i) / static_cast<double>(n);
          double y = static_cast<double>(j) / static_cast<double>(n);
          double z = static_cast<double>(k) / static_cast<double>(n);
          double a = static_cast<double>(l) / static_cast<double>(n);

          double dx = x
                      * (a * wbar(n + i - l) + y * wbar(n + i - j)
                         + z * wbar(n + i - k));
          double dy = y
                      * (a * wbar(n + j - l) + z * wbar(n + j - k)
                         + x * wbar(n + j - i));
          double dz = z
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
