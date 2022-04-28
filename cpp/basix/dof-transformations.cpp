// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-transformations.h"
#include "math.h"
#include "polyset.h"
#include <algorithm>
#include <xtensor/xview.hpp>

using namespace basix;
typedef std::map<
    cell::type,
    std::vector<std::tuple<
        std::function<xt::xtensor<double, 1>(const xt::xtensor<double, 1>&)>,
        xt::xtensor<double, 2>, double, xt::xtensor<double, 2>>>>
    mapinfo_t;

namespace
{
//-----------------------------------------------------------------------------
int find_first_subentity(cell::type cell_type, cell::type entity_type)
{
  const int edim = cell::topological_dimension(entity_type);
  std::vector<cell::type> entities = cell::subentity_types(cell_type)[edim];
  for (std::size_t i = 0; i < entities.size(); ++i)
  {
    if (entities[i] == entity_type)
      return i;
  }
  throw std::runtime_error("Entity not found");
}
//-----------------------------------------------------------------------------
void pull_back(maps::type map_type, xt::xtensor<double, 2>& u,
               const xt::xtensor<double, 2>& U, const xt::xtensor<double, 2>& J,
               const double detJ, const xt::xtensor<double, 2>& K)
{
  switch (map_type)
  {
  case maps::type::identity:
    u.assign(U);
    return;
  case maps::type::covariantPiola:
    maps::covariant_piola(u, U, K, 1.0 / detJ, J);
    return;
  case maps::type::contravariantPiola:
    maps::contravariant_piola(u, U, K, 1.0 / detJ, J);
    return;
  case maps::type::doubleCovariantPiola:
    maps::double_covariant_piola(u, U, K, 1.0 / detJ, J);
    return;
  case maps::type::doubleContravariantPiola:
    maps::double_contravariant_piola(u, U, K, 1.0 / detJ, J);
    return;
  default:
    throw std::runtime_error("Map not implemented");
  }
}
//-----------------------------------------------------------------------------
mapinfo_t get_mapinfo(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::point:
    return mapinfo_t();
  case cell::type::interval:
    return mapinfo_t();
  case cell::type::triangle:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[1], pt[0]});
      };
      const xt::xtensor<double, 2> J = {{0., 1.}, {1., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K = {{0., 1.}, {1., 0.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    return mapinfo;
  }
  case cell::type::quadrilateral:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[0], pt[1]});
      };
      const xt::xtensor<double, 2> J = {{-1., 0.}, {0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K = {{-1., 0.}, {0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    return mapinfo;
  }
  case cell::type::tetrahedron:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    mapinfo[cell::type::triangle] = {};
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[0], pt[2], pt[1]});
      };
      const xt::xtensor<double, 2> J
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[2], pt[0], pt[1]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 0., 1.}, {1., 0., 0.}, {0., 1., 0.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {0., 0., 1.}, {1., 0., 0.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[0], pt[2], pt[1]});
      };
      const xt::xtensor<double, 2> J
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    return mapinfo;
  }
  case cell::type::hexahedron:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    mapinfo[cell::type::quadrilateral] = {};
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[0], pt[1], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[1], pt[0], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {-1., 0., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::quadrilateral].push_back(
          std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[1], pt[0], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::quadrilateral].push_back(
          std::make_tuple(map, J, detJ, K));
    }
    return mapinfo;
  }
  case cell::type::prism:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    mapinfo[cell::type::triangle] = {};
    mapinfo[cell::type::quadrilateral] = {};
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[0], pt[1], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[1] - pt[0], pt[0], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{-1., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {-1., -1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[1], pt[0], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[2], pt[1], pt[0]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 0., -1.}, {0., 1., 0.}, {1., 0., 0.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}};
      mapinfo[cell::type::quadrilateral].push_back(
          std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[2], pt[1], pt[0]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
      mapinfo[cell::type::quadrilateral].push_back(
          std::make_tuple(map, J, detJ, K));
    }
    return mapinfo;
  }
  case cell::type::pyramid:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    mapinfo[cell::type::triangle] = {};
    mapinfo[cell::type::quadrilateral] = {};
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[0], pt[1], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[1], pt[0], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {-1., 0., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::quadrilateral].push_back(
          std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[1], pt[0], pt[2]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::quadrilateral].push_back(
          std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({1 - pt[2] - pt[0], pt[1], pt[0]});
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., -1.}, {0., 1., 0.}, {1., 0., 0.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., -1.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[2], pt[1], pt[0]});
      };
      const xt::xtensor<double, 2> J
          = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    return mapinfo;
  }
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> compute_transformation(
    cell::type cell_type,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const xt::xtensor<double, 2> J,
    const double detJ, const xt::xtensor<double, 2> K,
    const std::function<xt::xtensor<double, 1>(const xt::xtensor<double, 1>&)>
        map_point,
    const int degree, const int tdim, const int entity, const int vs,
    const maps::type map_type)
{
  if (x[tdim].size() == 0 or x[tdim][entity].shape(0) == 0)
    return xt::xtensor<double, 2>({0, 0});

  const xt::xtensor<double, 2>& pts = x[tdim][entity];
  const xt::xtensor<double, 3>& imat = M[tdim][entity];

  const std::size_t ndofs = imat.shape(0);
  const std::size_t npts = pts.shape(0);
  const int psize = polyset::dim(cell_type, degree);

  std::size_t dofstart = 0;
  for (int d = 0; d < tdim; ++d)
    for (std::size_t i = 0; i < M[d].size(); ++i)
      dofstart += M[d][i].shape(0);
  for (int i = 0; i < entity; ++i)
    dofstart += M[tdim][i].shape(0);

  std::size_t total_ndofs = 0;
  for (int d = 0; d <= 3; ++d)
    for (std::size_t i = 0; i < M[d].size(); ++i)
      total_ndofs += M[d][i].shape(0);

  // Map the points to reverse the edge, then tabulate at those points
  xt::xtensor<double, 2> mapped_pts(pts.shape());
  for (std::size_t p = 0; p < mapped_pts.shape(0); ++p)
    xt::row(mapped_pts, p) = map_point(xt::row(pts, p));

  xt::xtensor<double, 2> polyset_vals
      = xt::view(polyset::tabulate(cell_type, degree, 0, mapped_pts), 0,
                 xt::all(), xt::all());
  xt::xtensor<double, 3> tabulated_data(
      {npts, total_ndofs, static_cast<std::size_t>(vs)});

  for (int j = 0; j < vs; ++j)
  {
    auto data_view = xt::view(tabulated_data, xt::all(), xt::all(), j);
    xt::xtensor<double, 2> C
        = xt::view(coeffs, xt::all(), xt::range(psize * j, psize * j + psize));
    xt::xtensor<double, 2> Ct = xt::transpose(C);
    auto result = math::dot(polyset_vals, Ct);
    data_view.assign(result);
  }

  // Pull back
  xt::xtensor<double, 3> pulled_data(tabulated_data.shape());

  xt::xtensor<double, 2> temp_data(
      {pulled_data.shape(1), pulled_data.shape(2)});
  for (std::size_t i = 0; i < npts; ++i)
  {
    pull_back(map_type, temp_data,
              xt::view(tabulated_data, i, xt::all(), xt::all()), J, detJ, K);
    auto pview = xt::view(pulled_data, i, xt::all(), xt::all());
    pview.assign(temp_data);
  }

  // Interpolate to calculate coefficients
  xt::xtensor<double, 3> dof_data = xt::view(
      pulled_data, xt::all(), xt::range(dofstart, dofstart + ndofs), xt::all());
  xt::xtensor<double, 2> transform = xt::zeros<double>({ndofs, ndofs});
  for (int i = 0; i < vs; ++i)
  {
    xt::xtensor<double, 2> mat = xt::view(imat, xt::all(), i, xt::all());
    xt::xtensor<double, 2> values = xt::view(dof_data, xt::all(), xt::all(), i);

    transform += xt::transpose(math::dot(mat, values));
  }
  return transform;
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
std::map<cell::type, xt::xtensor<double, 3>>
doftransforms::compute_entity_transformations(
    cell::type cell_type,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const int degree, const int vs,
    maps::type map_type)
{
  std::map<cell::type, xt::xtensor<double, 3>> out;

  mapinfo_t mapinfo = get_mapinfo(cell_type);
  for (auto item : mapinfo)
  {
    const int tdim = cell::topological_dimension(item.first);
    const int entity = find_first_subentity(cell_type, item.first);
    const std::size_t ndofs
        = M[tdim].size() == 0 ? 0 : M[tdim][entity].shape(0);
    xt::xtensor<double, 3> transform({item.second.size(), ndofs, ndofs});
    for (std::size_t i = 0; i < item.second.size(); ++i)
    {
      auto [map, J, detJ, K] = item.second[i];
      auto t = xt::view(transform, i, xt::all(), xt::all());
      auto t2 = compute_transformation(cell_type, x, M, coeffs, J, detJ, K, map,
                                       degree, tdim, entity, vs, map_type);
      t.assign(t2);
    }
    out[item.first] = transform;
  }

  return out;
}
//-----------------------------------------------------------------------------
