// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "quadrature.h"
#include <map>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
template <typename U, typename V>
xt::xtensor_fixed<double, xt::xshape<3>> cross3(const U& u, const V& v)
{
  xt::xtensor_fixed<double, xt::xshape<3>> c;
  c[0] = u[1] * v[2] - u[2] * v[1];
  c[1] = u[2] * v[0] - u[0] * v[2];
  c[2] = u[0] * v[1] - u[1] * v[0];
  return c;
}
} // namespace

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> cell::geometry(cell::type celltype)
{
  switch (celltype)
  {
  case cell::type::interval:
    return xt::xtensor<double, 2>({{0.0}, {1.0}});
  case cell::type::triangle:
    return xt::xtensor<double, 2>({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
  case cell::type::quadrilateral:
    return xt::xtensor<double, 2>(
        {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});
  case cell::type::tetrahedron:
    return xt::xtensor<double, 2>(
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  case cell::type::prism:
    return xt::xtensor<double, 2>({{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 0.0, 1.0},
                                   {1.0, 0.0, 1.0},
                                   {0.0, 1.0, 1.0}});
  case cell::type::pyramid:
    return xt::xtensor<double, 2>({{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {1.0, 1.0, 0.0},
                                   {0.0, 0.0, 1.0}});
  case cell::type::hexahedron:
    return xt::xtensor<double, 2>({{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {1.0, 1.0, 0.0},
                                   {0.0, 0.0, 1.0},
                                   {1.0, 0.0, 1.0},
                                   {0.0, 1.0, 1.0},
                                   {1.0, 1.0, 1.0}});
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::vector<int>>> cell::topology(cell::type celltype)
{
  std::vector<std::vector<std::vector<int>>> t;

  switch (celltype)
  {
  case cell::type::interval:
    t.resize(2);
    // Vertices
    t[0] = {{0}, {1}};
    // Cell
    t[1] = {{0, 1}};
    break;
  case cell::type::triangle:
    t.resize(3);
    // Vertices
    t[0] = {{0}, {1}, {2}};
    // Edges
    t[1] = {{1, 2}, {0, 2}, {0, 1}};
    // Cell
    t[2] = {{0, 1, 2}};
    break;
  case cell::type::quadrilateral:
    t.resize(3);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    // Cell
    t[2] = {{0, 1, 2, 3}};
    break;
  case cell::type::tetrahedron:
    t.resize(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}};
    // Edges
    t[1] = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
    // Faces
    t[2] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
    // Cell
    t[3] = {{0, 1, 2, 3}};
    break;
  case cell::type::prism:
    t.resize(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}, {4}, {5}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 4},
            {2, 5}, {3, 4}, {3, 5}, {4, 5}};
    // Faces
    t[2] = {{0, 1, 2}, {0, 1, 3, 4}, {0, 2, 3, 5}, {1, 2, 4, 5}, {3, 4, 5}};
    // Cell
    t[3] = {{0, 1, 2, 3, 4, 5}};
    break;
  case cell::type::pyramid:
    t.resize(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}, {4}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    // Faces
    t[2] = {{0, 1, 2, 3}, {0, 1, 4}, {0, 2, 4}, {1, 3, 4}, {2, 3, 4}};
    // Cell
    t[3] = {{0, 1, 2, 3, 4}};
    break;
  case cell::type::hexahedron:
    t.resize(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
            {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7}};
    // Faces
    t[2] = {{0, 1, 2, 3}, {0, 1, 4, 5}, {0, 2, 4, 6},
            {1, 3, 5, 7}, {2, 3, 6, 7}, {4, 5, 6, 7}};
    // Cell
    t[3] = {{0, 1, 2, 3, 4, 5, 6, 7}};
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }

  return t;
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
xt::xtensor<double, 2> cell::sub_entity_geometry(cell::type celltype, int dim,
                                                 int index)
{
  const std::vector<std::vector<std::vector<int>>> cell_topology
      = cell::topology(celltype);
  if (dim < 0 or dim >= (int)cell_topology.size())
    throw std::runtime_error("Invalid dimension for sub-entity");
  const xt::xtensor<double, 2> cell_geometry = cell::geometry(celltype);
  const std::vector<std::vector<int>>& t = cell_topology[dim];
  if (index < 0 or index >= (int)t.size())
    throw std::runtime_error("Invalid entity index");

  xt::xtensor<double, 2> sub_entity({t[index].size(), cell_geometry.shape(1)});
  for (std::size_t i = 0; i < sub_entity.shape(0); ++i)
    xt::row(sub_entity, i) = xt::row(cell_geometry, t[index][i]);
  return sub_entity;
}
//----------------------------------------------------------------------------
int cell::num_sub_entities(cell::type celltype, int dim)
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
  switch (t[dim][index].size())
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
const std::string& cell::type_to_str(cell::type type)
{
  static const std::map<cell::type, std::string> type_to_name
      = {{cell::type::point, "point"},
         {cell::type::interval, "interval"},
         {cell::type::triangle, "triangle"},
         {cell::type::tetrahedron, "tetrahedron"},
         {cell::type::quadrilateral, "quadrilateral"},
         {cell::type::pyramid, "pyramid"},
         {cell::type::prism, "prism"},
         {cell::type::hexahedron, "hexahedron"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
double cell::volume(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::interval:
    return 1;
  case cell::type::triangle:
    return 0.5;
  case cell::type::quadrilateral:
    return 1;
  case cell::type::tetrahedron:
    return 1.0 / 6;
  case cell::type::hexahedron:
    return 1;
  case cell::type::prism:
    return 0.5;
  case cell::type::pyramid:
    return 1.0 / 3;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> cell::facet_outward_normals(cell::type cell_type)
{
  xt::xtensor<double, 2> normals = cell::facet_normals(cell_type);
  const std::vector<bool> facet_orientations
      = cell::facet_orientations(cell_type);
  for (std::size_t f = 0; f < normals.shape(0); ++f)
  {
    if (facet_orientations[f])
      xt::row(normals, f) *= -1.0;
  }

  return normals;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> cell::facet_normals(cell::type cell_type)
{
  const int tdim = cell::topological_dimension(cell_type);
  const xt::xtensor<double, 2> x = cell::geometry(cell_type);
  const std::vector<std::vector<int>> facets
      = cell::topology(cell_type)[tdim - 1];
  xt::xtensor<double, 2> normals({facets.size(), (std::size_t)tdim});

  switch (tdim)
  {
  case 1:
    return xt::ones<double>({1, 1});
  case 2:
  {
    for (std::size_t f = 0; f < facets.size(); ++f)
    {
      const std::vector<int>& facet = facets[f];
      auto normal = xt::row(normals, f);
      assert(facet.size() == 2);
      normal(0) = x(facet[1], 1) - x(facet[0], 1);
      normal(1) = x(facet[0], 0) - x(facet[1], 0);
      normal /= xt::sqrt(xt::sum(normal * normal));
    }
    return normals;
  }
  case 3:
  {
    for (std::size_t f = 0; f < facets.size(); ++f)
    {
      const std::vector<int>& facet = facets[f];
      auto normal = xt::row(normals, f);

      assert(facets[f].size() == 3 or facets[f].size() == 4);
      auto e0 = xt::row(x, facet[1]) - xt::row(x, facet[0]);
      auto e1 = xt::row(x, facet[2]) - xt::row(x, facet[0]);
      normal = cross3(e0, e1);
      normal /= xt::sqrt(xt::sum(normal * normal));
    }
    return normals;
  }
  default:
    throw std::runtime_error("Wrong topological dimension");
  }
}
//-----------------------------------------------------------------------------
std::vector<bool> cell::facet_orientations(cell::type cell_type)
{
  const std::size_t tdim = cell::topological_dimension(cell_type);
  const xt::xtensor<double, 2> x = cell::geometry(cell_type);
  const std::vector<std::vector<int>> facets
      = cell::topology(cell_type)[tdim - 1];

  const xt::xtensor<double, 2> normals = cell::facet_normals(cell_type);
  const xt::xtensor<double, 1> midpoint = xt::mean(x, 0);
  std::vector<bool> orientations(normals.shape(0));
  for (std::size_t f = 0; f < normals.shape(0); ++f)
  {
    auto normal = xt::row(normals, f);
    auto x0 = xt::row(x, facets[f][0]) - midpoint;
    const double dot = xt::sum(x0 * normal)();
    orientations[f] = dot < 0;
  }

  return orientations;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1> cell::facet_reference_volumes(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::interval:
    return {0, 0};
  case cell::type::triangle:
    return {cell::volume(cell::type::interval),
            cell::volume(cell::type::interval),
            cell::volume(cell::type::interval)};
  case cell::type::quadrilateral:
    return {
        cell::volume(cell::type::interval), cell::volume(cell::type::interval),
        cell::volume(cell::type::interval), cell::volume(cell::type::interval)};
  case cell::type::tetrahedron:
    return {
        cell::volume(cell::type::triangle), cell::volume(cell::type::triangle),
        cell::volume(cell::type::triangle), cell::volume(cell::type::triangle)};
  case cell::type::hexahedron:
    return {cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral)};
  case cell::type::prism:
    return {cell::volume(cell::type::triangle),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::quadrilateral),
            cell::volume(cell::type::triangle)};
  case cell::type::pyramid:
    return {
        cell::volume(cell::type::quadrilateral),
        cell::volume(cell::type::triangle), cell::volume(cell::type::triangle),
        cell::volume(cell::type::triangle), cell::volume(cell::type::triangle)};
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> cell::facet_jacobians(cell::type cell_type)
{
  const std::size_t tdim = cell::topological_dimension(cell_type);
  const xt::xtensor<double, 2> x = cell::geometry(cell_type);
  const std::vector<std::vector<int>> facets
      = cell::topology(cell_type)[tdim - 1];
  xt::xtensor<double, 3> jacobians({facets.size(), tdim, tdim - 1});
  if (tdim == 2 or tdim == 3)
  {
    for (std::size_t f = 0; f < facets.size(); ++f)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        for (std::size_t i = 0; i < tdim; ++i)
          jacobians(f, i, j) = x(facets[f][1 + j], i) - x(facets[f][0], i);
  }
  else
  {
    throw std::runtime_error(
        "Facet jacobians not supported for this cell type.");
  }

  return jacobians;
}
//-----------------------------------------------------------------------------
