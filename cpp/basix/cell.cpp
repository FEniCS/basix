// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "math.h"
#include "mdspan.hpp"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> cell::geometry(cell::type celltype)
{
  switch (celltype)
  {
  case cell::type::point:
    return xt::xtensor<double, 2>({{}});
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
std::pair<std::vector<double>, std::array<std::size_t, 2>>
cell::geometry_new(cell::type celltype)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{}, {0, 1}};
  case cell::type::interval:
    return {{0.0, 1.0}, {2, 1}};
  case cell::type::triangle:
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0}, {3, 2}};
    // return xt::xtensor<double, 2>({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
  case cell::type::quadrilateral:
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0}, {4, 2}};
    // return xt::xtensor<double, 2>(
    //     {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});
  case cell::type::tetrahedron:
    return {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
            {4, 3}};
  case cell::type::prism:
    return {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
             0.0, 1.0, 0.0, 1.0, 1.0},
            {6, 3}};
  case cell::type::pyramid:
    return {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
             0.0, 1.0},
            {5, 3}};
  case cell::type::hexahedron:
    return {{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
             0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {8, 3}};
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::vector<int>>> cell::topology(cell::type celltype)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{{0}}};
  case cell::type::interval:
    return {{{0}, {1}}, {{0, 1}}};
  case cell::type::triangle:
  {
    std::vector<std::vector<std::vector<int>>> t(3);
    // Vertices
    t[0] = {{0}, {1}, {2}};
    // Edges
    t[1] = {{1, 2}, {0, 2}, {0, 1}};
    // Cell
    t[2] = {{0, 1, 2}};
    return t;
  }
  case cell::type::quadrilateral:
  {
    std::vector<std::vector<std::vector<int>>> t(3);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    // Cell
    t[2] = {{0, 1, 2, 3}};
    return t;
  }
  case cell::type::tetrahedron:
  {
    std::vector<std::vector<std::vector<int>>> t(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}};
    // Edges
    t[1] = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
    // Faces
    t[2] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
    // Cell
    t[3] = {{0, 1, 2, 3}};
    return t;
  }
  case cell::type::prism:
  {
    std::vector<std::vector<std::vector<int>>> t(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}, {4}, {5}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 4},
            {2, 5}, {3, 4}, {3, 5}, {4, 5}};
    // Faces
    t[2] = {{0, 1, 2}, {0, 1, 3, 4}, {0, 2, 3, 5}, {1, 2, 4, 5}, {3, 4, 5}};
    // Cell
    t[3] = {{0, 1, 2, 3, 4, 5}};
    return t;
  }
  case cell::type::pyramid:
  {
    std::vector<std::vector<std::vector<int>>> t(4);
    // Vertices
    t[0] = {{0}, {1}, {2}, {3}, {4}};
    // Edges
    t[1] = {{0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    // Faces
    t[2] = {{0, 1, 2, 3}, {0, 1, 4}, {0, 2, 4}, {1, 3, 4}, {2, 3, 4}};
    // Cell
    t[3] = {{0, 1, 2, 3, 4}};
    return t;
  }
  case cell::type::hexahedron:
  {
    std::vector<std::vector<std::vector<int>>> t(4);
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
    return t;
  }
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::vector<std::vector<int>>>>
cell::sub_entity_connectivity(cell::type celltype)
{
  switch (celltype)
  {
  case cell::type::point:
    return {{{{0}}}};
  case cell::type::interval:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(2);
    // Vertices
    t[0] = {{{0}, {0}}, {{1}, {0}}};
    // Edge
    t[1] = {{{0, 1}, {0}}};
    return t;
  }
  case cell::type::triangle:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(3);
    // Vertices
    t[0] = {{{0}, {1, 2}, {0}}, {{1}, {0, 2}, {0}}, {{2}, {0, 1}, {0}}};
    // Edges
    t[1] = {{{1, 2}, {0}, {0}}, {{0, 2}, {1}, {0}}, {{0, 1}, {2}, {0}}};
    // Face
    t[2] = {{{0, 1, 2}, {0, 1, 2}, {0}}};
    return t;
  }
  case cell::type::quadrilateral:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(3);
    // Vertices
    t[0] = {{{0}, {0, 1}, {0}},
            {{1}, {0, 2}, {0}},
            {{2}, {1, 3}, {0}},
            {{3}, {2, 3}, {0}}};
    // Edges
    t[1] = {{{0, 1}, {0}, {0}},
            {{0, 2}, {1}, {0}},
            {{1, 3}, {2}, {0}},
            {{2, 3}, {3}, {0}}};
    // Face
    t[2] = {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0}}};
    return t;
  }
  case cell::type::tetrahedron:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(4);
    // Vertices
    t[0] = {{{0}, {3, 4, 5}, {1, 2, 3}, {0}},
            {{1}, {1, 2, 5}, {0, 2, 3}, {0}},
            {{2}, {0, 2, 4}, {0, 1, 3}, {0}},
            {{3}, {0, 1, 3}, {0, 1, 2}, {0}}};
    // Edges
    t[1] = {
        {{2, 3}, {0}, {0, 1}, {0}}, {{1, 3}, {1}, {0, 2}, {0}},
        {{1, 2}, {2}, {0, 3}, {0}}, {{0, 3}, {3}, {1, 2}, {0}},
        {{0, 2}, {4}, {1, 3}, {0}}, {{0, 1}, {5}, {2, 3}, {0}},
    };
    // Faces
    t[2] = {{{1, 2, 3}, {0, 1, 2}, {0}, {0}},
            {{0, 2, 3}, {0, 3, 4}, {1}, {0}},
            {{0, 1, 3}, {1, 3, 5}, {2}, {0}},
            {{0, 1, 2}, {2, 4, 5}, {3}, {0}}};
    // Volume
    t[3] = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3}, {0}}};
    return t;
  }
  case cell::type::hexahedron:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(4);
    // Vertices
    t[0] = {
        {{0}, {0, 1, 2}, {0, 1, 2}, {0}},  {{1}, {0, 3, 4}, {0, 1, 3}, {0}},
        {{2}, {1, 5, 6}, {0, 2, 4}, {0}},  {{3}, {3, 5, 7}, {0, 3, 4}, {0}},
        {{4}, {2, 8, 9}, {1, 2, 5}, {0}},  {{5}, {4, 8, 10}, {1, 3, 5}, {0}},
        {{6}, {6, 9, 11}, {2, 4, 5}, {0}}, {{7}, {7, 10, 11}, {3, 4, 5}, {0}}};
    // Edges
    t[1] = {{{0, 1}, {0}, {0, 1}, {0}},  {{0, 2}, {1}, {0, 2}, {0}},
            {{0, 4}, {2}, {1, 2}, {0}},  {{1, 3}, {3}, {0, 3}, {0}},
            {{1, 5}, {4}, {1, 3}, {0}},  {{2, 3}, {5}, {0, 4}, {0}},
            {{2, 6}, {6}, {2, 4}, {0}},  {{3, 7}, {7}, {3, 4}, {0}},
            {{4, 5}, {8}, {1, 5}, {0}},  {{4, 6}, {9}, {2, 5}, {0}},
            {{5, 7}, {10}, {3, 5}, {0}}, {{6, 7}, {11}, {4, 5}, {0}}};
    // Faces
    t[2] = {{{0, 1, 2, 3}, {0, 1, 3, 5}, {0}, {0}},
            {{0, 1, 4, 5}, {0, 2, 4, 8}, {1}, {0}},
            {{0, 2, 4, 6}, {1, 2, 6, 9}, {2}, {0}},
            {{1, 3, 5, 7}, {3, 4, 7, 10}, {3}, {0}},
            {{2, 3, 6, 7}, {5, 6, 7, 11}, {4}, {0}},
            {{4, 5, 6, 7}, {8, 9, 10, 11}, {5}, {0}}};
    // Volume
    t[3] = {{{0, 1, 2, 3, 4, 5, 6, 7},
             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
             {0, 1, 2, 3, 4, 5},
             {0}}};
    return t;
  }
  case cell::type::prism:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(4);
    // Vertices
    t[0] = {{{0}, {0, 1, 2}, {0, 1, 2}, {0}}, {{1}, {0, 3, 4}, {0, 1, 3}, {0}},
            {{2}, {1, 3, 5}, {0, 2, 3}, {0}}, {{3}, {2, 6, 7}, {1, 2, 4}, {0}},
            {{4}, {4, 6, 8}, {1, 3, 4}, {0}}, {{5}, {5, 7, 8}, {2, 3, 4}, {0}}};
    // Edges
    t[1] = {{{0, 1}, {0}, {0, 1}, {0}}, {{0, 2}, {1}, {0, 2}, {0}},
            {{0, 3}, {2}, {1, 2}, {0}}, {{1, 2}, {3}, {0, 3}, {0}},
            {{1, 4}, {4}, {1, 3}, {0}}, {{2, 5}, {5}, {2, 3}, {0}},
            {{3, 4}, {6}, {1, 4}, {0}}, {{3, 5}, {7}, {2, 4}, {0}},
            {{4, 5}, {8}, {3, 4}, {0}}};
    // Faces
    t[2] = {{{0, 1, 2}, {0, 1, 3}, {0}, {0}},
            {{0, 1, 3, 4}, {0, 2, 4, 6}, {1}, {0}},
            {{0, 2, 3, 5}, {1, 2, 5, 7}, {2}, {0}},
            {{1, 2, 4, 5}, {3, 4, 5, 8}, {3}, {0}},
            {{3, 4, 5}, {6, 7, 8}, {4}, {0}}};
    // Volume
    t[3] = {{{0, 1, 2, 3, 4, 5},
             {0, 1, 2, 3, 4, 5, 6, 7, 8},
             {0, 1, 2, 3, 4},
             {0}}};
    return t;
  }
  case cell::type::pyramid:
  {
    std::vector<std::vector<std::vector<std::vector<int>>>> t(4);
    // Vertices
    t[0] = {{{0}, {0, 1, 2}, {0, 1, 2}, {0}},
            {{1}, {0, 3, 4}, {0, 1, 3}, {0}},
            {{2}, {1, 5, 6}, {0, 2, 4}, {0}},
            {{3}, {3, 5, 7}, {0, 3, 4}, {0}},
            {{4}, {2, 4, 6, 7}, {1, 2, 3, 4}, {0}}};
    // Edges
    t[1] = {{{0, 1}, {0}, {0, 1}, {0}}, {{0, 2}, {1}, {0, 2}, {0}},
            {{0, 4}, {2}, {1, 2}, {0}}, {{1, 3}, {3}, {0, 3}, {0}},
            {{1, 4}, {4}, {1, 3}, {0}}, {{2, 3}, {5}, {0, 4}, {0}},
            {{2, 4}, {6}, {2, 4}, {0}}, {{3, 4}, {7}, {3, 4}, {0}}};
    // Faces
    t[2] = {{{0, 1, 2, 3}, {0, 1, 3, 5}, {0}, {0}},
            {{0, 1, 4}, {0, 2, 4}, {1}, {0}},
            {{0, 2, 4}, {1, 2, 6}, {2}, {0}},
            {{1, 3, 4}, {3, 4, 7}, {3}, {0}},
            {{2, 3, 4}, {5, 6, 7}, {4}, {0}}};
    // Volume
    t[3] = {{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4}, {0}}};
    return t;
  }
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
int cell::topological_dimension(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::point:
    return 0;
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
std::pair<std::vector<double>, std::array<std::size_t, 2>>
cell::sub_entity_geometry_new(cell::type celltype, int dim, int index)
{
  const std::vector<std::vector<std::vector<int>>> cell_topology
      = cell::topology(celltype);
  if (dim < 0 or dim >= (int)cell_topology.size())
    throw std::runtime_error("Invalid dimension for sub-entity");

  const auto [cell_geometry, shape] = cell::geometry_new(celltype);
  const std::vector<std::vector<int>>& t = cell_topology[dim];
  if (index < 0 or index >= (int)t.size())
    throw std::runtime_error("Invalid entity index");

  using cmdspan2_t
      = std::experimental::mdspan<const double,
                                  std::experimental::dextents<std::size_t, 2>>;
  cmdspan2_t geometry(cell_geometry.data(), shape);

  std::array<std::size_t, 2> subshape = {t[index].size(), geometry.extent(1)};
  std::vector<double> sub_geometry(subshape[0] * subshape[1]);
  using mdspan2_t
      = std::experimental::mdspan<double,
                                  std::experimental::dextents<std::size_t, 2>>;
  mdspan2_t sub_entity(sub_geometry.data(), subshape);
  for (std::size_t i = 0; i < sub_entity.extent(0); ++i)
    for (std::size_t j = 0; j < sub_entity.extent(1); ++j)
      sub_entity(i, j) = geometry(t[index][i], j);

  return {sub_geometry, subshape};
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
double cell::volume(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::point:
    return 0;
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
std::pair<std::vector<double>, std::array<std::size_t, 2>>
cell::facet_outward_normals(cell::type cell_type)
{
  auto [normals, shape] = cell::facet_normals(cell_type);
  using mdspan2_t
      = std::experimental::mdspan<double,
                                  std::experimental::dextents<std::size_t, 2>>;
  mdspan2_t n(normals.data(), shape);
  const std::vector<bool> facet_orientations
      = cell::facet_orientations(cell_type);
  for (std::size_t f = 0; f < n.extent(0); ++f)
  {
    if (facet_orientations[f])
    {
      for (std::size_t k = 0; k < n.extent(1); ++k)
        n(f, k) *= -1.0;
    }
  }

  return {normals, shape};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
cell::facet_normals(cell::type cell_type)
{
  const std::size_t tdim = cell::topological_dimension(cell_type);
  const xt::xtensor<double, 2> x = cell::geometry(cell_type);
  const std::vector<std::vector<int>> facets
      = cell::topology(cell_type)[tdim - 1];

  using mdspan2_t
      = std::experimental::mdspan<double,
                                  std::experimental::dextents<std::size_t, 2>>;
  std::array<std::size_t, 2> shape = {facets.size(), tdim};
  std::vector<double> normal(shape[0] * shape[1]);
  mdspan2_t n(normal.data(), shape);

  switch (tdim)
  {
  case 1:
    std::fill(normal.begin(), normal.end(), 1.0);
    return {normal, shape};
  case 2:
  {
    for (std::size_t f = 0; f < facets.size(); ++f)
    {
      const std::vector<int>& facet = facets[f];
      assert(facet.size() == 2);
      n(f, 0) = x(facet[1], 1) - x(facet[0], 1);
      n(f, 1) = x(facet[0], 0) - x(facet[1], 0);
      double L = std::sqrt(n(f, 0) * n(f, 0) + n(f, 1) * n(f, 1));
      n(f, 0) /= L;
      n(f, 1) /= L;
    }
    return {normal, shape};
  }
  case 3:
  {
    for (std::size_t f = 0; f < facets.size(); ++f)
    {
      const std::vector<int>& facet = facets[f];
      assert(facets[f].size() == 3 or facets[f].size() == 4);
      auto e0 = xt::row(x, facet[1]) - xt::row(x, facet[0]);
      auto e1 = xt::row(x, facet[2]) - xt::row(x, facet[0]);
      auto n_f = basix::math::cross(e0, e1);
      n_f /= xt::sqrt(xt::sum(n_f * n_f));
      for (std::size_t i = 0; i < 3; ++i)
        n(f, i) = n_f[i];
    }
    return {normal, shape};
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

  const auto [normals, shape] = cell::facet_normals(cell_type);
  using mdspan2_t
      = std::experimental::mdspan<const double,
                                  std::experimental::dextents<std::size_t, 2>>;
  mdspan2_t n(normals.data(), shape);

  const xt::xtensor<double, 1> midpoint = xt::mean(x, 0);
  std::vector<bool> orientations(n.extent(0));
  for (std::size_t f = 0; f < n.extent(0); ++f)
  {
    double dot = 0.0;
    for (std::size_t i = 0; i < n.extent(1); ++i)
      dot += n(facets[f][0], i) * (x(facets[f][0], i) - midpoint[i]);
    orientations[f] = dot < 0;
  }

  return orientations;
}
//-----------------------------------------------------------------------------
std::vector<double> cell::facet_reference_volumes(cell::type cell_type)
{
  const int tdim = cell::topological_dimension(cell_type);
  const std::vector<cell::type> facet_types
      = cell::subentity_types(cell_type)[tdim - 1];

  std::vector<double> out;
  for (auto& facet_type : facet_types)
    out.push_back(cell::volume(facet_type));

  return out;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<cell::type>> cell::subentity_types(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::point:
    return {{cell::type::point}};
  case cell::type::interval:
    return {{cell::type::point, cell::type::point}, {cell::type::interval}};
  case cell::type::triangle:
    return {{cell::type::point, cell::type::point, cell::type::point},
            {cell::type::interval, cell::type::interval, cell::type::interval},
            {cell::type::triangle}};
  case cell::type::quadrilateral:
    return {{cell::type::point, cell::type::point, cell::type::point,
             cell::type::point},
            {cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval},
            {cell::type::quadrilateral}};
  case cell::type::tetrahedron:
    return {{cell::type::point, cell::type::point, cell::type::point,
             cell::type::point},
            {cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval},
            {cell::type::triangle, cell::type::triangle, cell::type::triangle,
             cell::type::triangle},
            {cell::type::tetrahedron}};
  case cell::type::hexahedron:
    return {{cell::type::point, cell::type::point, cell::type::point,
             cell::type::point, cell::type::point, cell::type::point,
             cell::type::point, cell::type::point},
            {cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval},
            {cell::type::quadrilateral, cell::type::quadrilateral,
             cell::type::quadrilateral, cell::type::quadrilateral,
             cell::type::quadrilateral, cell::type::quadrilateral},
            {cell::type::hexahedron}};
  case cell::type::prism:
    return {{cell::type::point, cell::type::point, cell::type::point,
             cell::type::point, cell::type::point, cell::type::point},
            {cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval},
            {cell::type::triangle, cell::type::quadrilateral,
             cell::type::quadrilateral, cell::type::quadrilateral,
             cell::type::triangle},
            {cell::type::prism}};
  case cell::type::pyramid:
    return {{cell::type::point, cell::type::point, cell::type::point,
             cell::type::point, cell::type::point},
            {cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval, cell::type::interval,
             cell::type::interval, cell::type::interval},
            {cell::type::quadrilateral, cell::type::triangle,
             cell::type::triangle, cell::type::triangle, cell::type::triangle},
            {cell::type::pyramid}};
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 3>>
cell::facet_jacobians(cell::type cell_type)
{
  const std::size_t tdim = cell::topological_dimension(cell_type);
  if (tdim != 2 and tdim != 3)
  {
    throw std::runtime_error(
        "Facet jacobians not supported for this cell type.");
  }

  const xt::xtensor<double, 2> x = cell::geometry(cell_type);
  const std::vector<std::vector<int>> facets
      = cell::topology(cell_type)[tdim - 1];

  std::array<std::size_t, 3> shape = {facets.size(), tdim, tdim - 1};
  std::vector<double> jacobians(shape[0] * shape[1] * shape[2]);
  using mdspan3_t
      = std::experimental::mdspan<double,
                                  std::experimental::dextents<std::size_t, 3>>;
  mdspan3_t J(jacobians.data(), shape);

  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    const std::vector<int>& facet = facets[f];
    auto x0 = xt::row(x, facet[0]);
    for (std::size_t j = 0; j < tdim - 1; ++j)
      for (std::size_t k = 0; k < J.extent(1); ++k)
        J(f, k, j) = x(facet[1 + j], k) - x0[k];
  }

  return {std::move(jacobians), std::move(shape)};
}
//-----------------------------------------------------------------------------
