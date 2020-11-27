// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "lagrange.h"
#include "quadrature.h"
#include <iostream>
#include <map>

using namespace libtab;

//-----------------------------------------------------------------------------
Eigen::ArrayXXd cell::geometry(cell::type celltype)
{
  Eigen::ArrayXXd geom;
  switch (celltype)
  {
  case cell::type::interval:
    geom.resize(2, 1);
    geom << 0.0, 1.0;
    break;
  case cell::type::triangle:
    geom.resize(3, 2);
    geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
    break;
  case cell::type::quadrilateral:
    geom.resize(4, 2);
    geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    break;
  case cell::type::tetrahedron:
    geom.resize(4, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    break;
  case cell::type::prism:
    geom.resize(6, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 1.0;
    break;
  case cell::type::pyramid:
    geom.resize(5, 3);
    geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        0.0, 1.0;
    break;
  case cell::type::hexahedron:
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
std::vector<std::vector<std::vector<int>>> cell::topology(cell::type celltype)
{
  std::vector<std::vector<std::vector<int>>> topo;

  switch (celltype)
  {
  case cell::type::interval:
    topo.resize(2);
    // Vertices
    topo[0] = {{0}, {1}};
    // Cell
    topo[1] = {{0, 1}};
    break;
  case cell::type::triangle:
    topo.resize(3);
    // Vertices
    topo[0] = {{0}, {1}, {2}};
    // Edges
    topo[1] = {{1, 2}, {0, 2}, {0, 1}};
    // Cell
    topo[2] = {{0, 1, 2}};
    break;
  case cell::type::quadrilateral:
    topo.resize(3);
    // FIXME - check all these
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}};
    // Edges
    topo[1] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    // Cell
    topo[2] = {{0, 1, 2, 3}};
    break;
  case cell::type::tetrahedron:
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
  case cell::type::prism:
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
  case cell::type::pyramid:
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
  case cell::type::hexahedron:
    topo.resize(4);
    // FIXME: check over
    // Vertices
    topo[0] = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
    // Edges
    topo[1] = {{0, 1}, {0, 2}, {2, 3}, {1, 3}, {0, 4}, {1, 5},
               {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7}};
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
int cell::topological_dimension(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::interval:
    return 1;
  case cell::type::triangle:
    return 2;
  case cell::type::quadrilateral:
    return 2;
  case cell::type::tetrahedron:
    return 3;
  case cell::type::hexahedron:
    return 3;
  case cell::type::prism:
    return 3;
  case cell::type::pyramid:
    return 3;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
  return 0;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd cell::sub_entity_geometry(cell::type celltype, int dim,
                                          int index)
{
  std::vector<std::vector<std::vector<int>>> cell_topology
      = cell::topology(celltype);
  Eigen::ArrayXXd cell_geometry = cell::geometry(celltype);

  if (dim < 0 or dim >= (int)cell_topology.size())
    throw std::runtime_error("Invalid dimension for sub-entity");

  const std::vector<std::vector<int>>& t = cell_topology[dim];

  if (index < 0 or index >= (int)t.size())
    throw std::runtime_error("Invalid entity index");

  Eigen::ArrayXXd sub_entity(t[index].size(), cell_geometry.cols());
  for (int i = 0; i < sub_entity.rows(); ++i)
    sub_entity.row(i) = cell_geometry.row(t[index][i]);

  return sub_entity;
}
//----------------------------------------------------------------------------
int cell::sub_entity_count(cell::type celltype, int dim)
{
  const std::vector<std::vector<std::vector<int>>> cell_topology
      = cell::topology(celltype);
  return cell_topology.at(dim).size();
}
//----------------------------------------------------------------------------
cell::type cell::sub_entity_type(cell::type celltype, int dim, int index)
{
  const int tdim = cell::topological_dimension(celltype);
  assert(dim >= 0 and dim <= tdim);

  if (dim == 0)
    return cell::type::point;
  else if (dim == 1)
    return cell::type::interval;
  else if (dim == tdim)
    return celltype;

  const std::vector<std::vector<std::vector<int>>> t = cell::topology(celltype);
  const std::vector<int>& entity = t[dim][index];
  switch (entity.size())
  {
  case 3:
    return cell::type::triangle;
  case 4:
    return cell::type::quadrilateral;
  default:
    throw std::runtime_error("Error in sub_entity_type");
  }
}
//-----------------------------------------------------------------------------
cell::type cell::str_to_type(std::string name)
{
  static const std::map<std::string, cell::type> name_to_type
      = {{"point", cell::type::point},
         {"interval", cell::type::interval},
         {"triangle", cell::type::triangle},
         {"tetrahedron", cell::type::tetrahedron},
         {"quadrilateral", cell::type::quadrilateral},
         {"pyramid", cell::type::pyramid},
         {"prism", cell::type::prism},
         {"hexahedron", cell::type::hexahedron}};

  auto it = name_to_type.find(name);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find name " + name);

  return it->second;
}
//-----------------------------------------------------------------------------
