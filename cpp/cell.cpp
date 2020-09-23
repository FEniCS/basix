// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"

Cell::Cell(Cell::Type celltype) : _type(celltype)
{

  switch (_type)
  {
  case Cell::Type::interval:
    _geometry.resize(2, 1);
    _geometry << 0.0, 1.0;
    _topology.resize(2);
    // Vertices
    _topology[0].resize(2, 1);
    _topology[0] << 0, 1;
    // Cell
    _topology[1].resize(1, 2);
    _topology[1] << 0, 1;
    break;
  case Cell::Type::triangle:
    _geometry.resize(3, 2);
    _geometry << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
    _topology.resize(3);
    // Vertices
    _topology[0].resize(3, 1);
    _topology[0] << 0, 1, 2;
    // Edges
    _topology[1].resize(3, 2);
    _topology[1] << 1, 2, 2, 0, 0, 1;
    // Cell
    _topology[2].resize(1, 3);
    _topology[2] << 0, 1, 2;
    break;
  case Cell::Type::quadrilateral:
    _geometry.resize(4, 2);
    _geometry << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    _topology.resize(3);
    // FIXME - check all these
    // Vertices
    _topology[0].resize(4, 1);
    _topology[0] << 0, 1, 2, 3;
    // Edges
    _topology[1].resize(4, 2);
    _topology[1] << 1, 2, 2, 3, 3, 0, 0, 1;
    // Cell
    _topology[2].resize(1, 4);
    _topology[2] << 0, 1, 2, 3;
    break;
  case Cell::Type::tetrahedron:
    _geometry.resize(4, 3);
    _geometry << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    _topology.resize(4);
    // Vertices
    _topology[0].resize(4, 1);
    _topology[0] << 0, 1, 2, 3;
    // Edges
    _topology[1].resize(6, 2);
    _topology[1] << 2, 3, 1, 3, 1, 2, 0, 3, 0, 2, 0, 1;
    // Faces
    _topology[2].resize(4, 3);
    _topology[2] << 1, 2, 3, 2, 3, 0, 3, 0, 1, 0, 1, 2;
    // Cell
    _topology[3].resize(1, 4);
    _topology[3] << 0, 1, 2, 3;
    break;
  case Cell::Type::prism:
    _geometry.resize(6, 3);
    _geometry << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0, 1.0;
    _topology.resize(4);
    // Vertices
    _topology[0].resize(6, 1);
    _topology[0] << 0, 1, 2, 3, 4, 5;
    // Edges
    _topology[1].resize(9, 2);
    // Faces
    _topology[2].resize(5, 4);
    // Cell
    _topology[3].resize(1, 6);
    _topology[3] << 0, 1, 2, 3, 4, 5;
    break;
  case Cell::Type::pyramid:
    _geometry.resize(5, 3);
    _geometry << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0;
    _topology.resize(4);
    // Vertices
    _topology[0].resize(5, 1);
    _topology[0] << 0, 1, 2, 3, 4;
    // Edges
    _topology[1].resize(8, 2);
    // Faces
    _topology[2].resize(5, 4);
    // Cell
    _topology[3].resize(1, 5);
    _topology[3] << 0, 1, 2, 3, 4;
    break;
  case Cell::Type::hexahedron:
    _geometry.resize(8, 3);
    _geometry << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
int Cell::topological_dimension(Cell::Type cell_type)
{
  switch (cell_type)
  {
  case Cell::Type::interval:
    return 1;
  case Cell::Type::triangle:
    return 2;
  case Cell::Type::quadrilateral:
    return 2;
  case Cell::Type::tetrahedron:
    return 3;
  case Cell::Type::hexahedron:
    return 3;
  case Cell::Type::prism:
    return 3;
  case Cell::Type::pyramid:
    return 3;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
  return 0;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Cell::sub_entity_geometry(int dim, int index) const
{
  if (dim < 0 or dim >= (int)_topology.size())
    throw std::runtime_error("Invalid dimension for sub-entity");

  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& t
      = _topology[dim];

  if (index < 0 or index >= t.rows())
    throw std::runtime_error("Invalid entity index");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      sub_entity(t.cols(), _geometry.cols());

  for (int i = 0; i < sub_entity.rows(); ++i)
    sub_entity.row(i) = _geometry.row(t(index, i));

  return sub_entity;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Cell::create_lattice(int n, bool exterior) const
{
  const int gdim = _geometry.cols();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points;

  if (_type == Cell::Type::quadrilateral)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        2, _geometry.cols());
    hs.row(1) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(0) = (_geometry.row(2) - _geometry.row(0)) / static_cast<double>(n);

    int b = (exterior == false) ? 1 : 0;
    int m = (n - 2 * b + 1);
    points.resize(m * m, gdim);
    int c = 0;
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - b; ++j)
        points.row(c++) = _geometry.row(0) + hs.row(0) * i + hs.row(1) * j;
  }
  else if (_type == Cell::Type::hexahedron)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        3, _geometry.cols());
    hs.row(2) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(1) = (_geometry.row(2) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(0) = (_geometry.row(4) - _geometry.row(0)) / static_cast<double>(n);
    int b = (exterior == false) ? 1 : 0;
    int m = (n - 2 * b + 1);
    points.resize(m * m * m, gdim);
    int c = 0;
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - b; ++j)
        for (int k = b; k < n + 1 - b; ++k)
          points.row(c++) = _geometry.row(0) + hs.row(0) * i + hs.row(1) * j
                            + hs.row(2) * k;
  }
  else if (_type == Cell::Type::interval)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        1, _geometry.cols());
    hs.row(0) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);

    int b = (exterior == false) ? 1 : 0;
    int m = (n - 2 * b + 1);

    points.resize(m, gdim);
    int c = 0;
    for (int i = b; i < n + 1 - b; ++i)
      points.row(c++) = _geometry.row(0) + hs.row(0) * i;
  }
  else if (_type == Cell::Type::triangle)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        2, _geometry.cols());
    hs.row(1) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(0) = (_geometry.row(2) - _geometry.row(0)) / static_cast<double>(n);

    int b = (exterior == false) ? 1 : 0;
    int m = (n - 3 * b + 1) * (n - 3 * b + 2) / 2;
    points.resize(m, gdim);
    int c = 0;
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        points.row(c++) = _geometry.row(0) + hs.row(1) * j + hs.row(0) * i;
  }
  else if (_type == Cell::Type::tetrahedron)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        3, _geometry.cols());
    hs.row(2) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(1) = (_geometry.row(2) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(0) = (_geometry.row(3) - _geometry.row(0)) / static_cast<double>(n);

    int b = (exterior == false) ? 1 : 0;
    int m = (n - 4 * b + 1) * (n - 4 * b + 2) * (n - 4 * b + 3) / 6;
    points.resize(m, gdim);
    int c = 0;
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        for (int k = b; k < n + 1 - i - j - b; ++k)
          points.row(c++) = _geometry.row(0) + hs.row(2) * k + hs.row(1) * j
                            + hs.row(0) * i;
  }
  else if (_type == Cell::Type::prism)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        3, _geometry.cols());
    hs.row(2) = (_geometry.row(3) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(1) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(0) = (_geometry.row(2) - _geometry.row(0)) / static_cast<double>(n);

    int b = (exterior == false) ? 1 : 0;
    int m = (n - 2 * b + 1) * (n - 3 * b + 1) * (n - 3 * b + 2) / 2;
    points.resize(m, gdim);
    int c = 0;
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        for (int k = b; k < n + 1 - b; ++k)
          points.row(c++) = _geometry.row(0) + hs.row(2) * k + hs.row(1) * j
                            + hs.row(0) * i;
  }
  else if (_type == Cell::Type::pyramid)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
        3, _geometry.cols());
    hs.row(2) = (_geometry.row(4) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(1) = (_geometry.row(1) - _geometry.row(0)) / static_cast<double>(n);
    hs.row(0) = (_geometry.row(2) - _geometry.row(0)) / static_cast<double>(n);

    if (exterior == false)
      throw std::runtime_error("not implemented in pyramid");

    int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
    points.resize(m, gdim);
    int c = 0;
    for (int k = 0; k < n + 1; ++k)
      for (int i = 0; i < n + 1 - k; ++i)
        for (int j = 0; j < n + 1 - k; ++j)
          points.row(c++) = _geometry.row(0) + hs.row(2) * k + hs.row(1) * j
                            + hs.row(0) * i;
  }
  else
    throw std::runtime_error("Unsupported cell for lattice");

  return points;
}
