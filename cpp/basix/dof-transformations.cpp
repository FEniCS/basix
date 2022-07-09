// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-transformations.h"
#include "math.h"
#include "polyset.h"
#include <algorithm>
#include <array>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

using namespace basix;

namespace stdex = std::experimental;
using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan4_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

typedef std::map<
    cell::type,
    std::vector<std::tuple<
        std::function<std::array<double, 3>(xtl::span<const double>)>,
        xt::xtensor<double, 2>, double, xt::xtensor<double, 2>>>>
    mapinfo_t;

namespace
{
//-----------------------------------------------------------------------------
template <typename U>
xt::xtensor<typename U::value_type, 2> mdspan_to_xtensor2(const U& x)
{
  auto e = x.extents();
  xt::xtensor<typename U::value_type, 2> y({e.extent(0), e.extent(1)});
  for (std::size_t k0 = 0; k0 < e.extent(0); ++k0)
    for (std::size_t k1 = 0; k1 < e.extent(1); ++k1)
      y(k0, k1) = x(k0, k1);
  return y;
}
//----------------------------------------------------------------------------
template <typename U>
xt::xtensor<double, 4> mdspan_to_xtensor4(const U& x)
{
  auto e = x.extents();
  xt::xtensor<double, 4> y(
      {e.extent(0), e.extent(1), e.extent(2), e.extent(3)});
  for (std::size_t k0 = 0; k0 < e.extent(0); ++k0)
    for (std::size_t k1 = 0; k1 < e.extent(1); ++k1)
      for (std::size_t k2 = 0; k2 < e.extent(2); ++k2)
        for (std::size_t k3 = 0; k3 < e.extent(3); ++k3)
          y(k0, k1, k2, k3) = x(k0, k1, k2, k3);

  return y;
}
//-----------------------------------------------------------------------------
std::array<std::vector<xt::xtensor<double, 2>>, 4>
to_xtensor(const std::array<std::vector<cmdspan2_t>, 4>& x)
{
  std::array<std::vector<xt::xtensor<double, 2>>, 4> _x;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      _x[i].push_back(mdspan_to_xtensor2(x[i][j]));

  return _x;
}
//----------------------------------------------------------------------------
std::array<std::vector<xt::xtensor<double, 4>>, 4>
to_xtensor(const std::array<std::vector<cmdspan4_t>, 4>& M)
{
  std::array<std::vector<xt::xtensor<double, 4>>, 4> _M;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      _M[i].push_back(mdspan_to_xtensor4(M[i][j]));

  return _M;
}
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
  using u_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using U_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using J_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using K_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;

  auto _u = u_t(u.data(), u.shape(0), u.shape(1));
  auto _U = U_t(U.data(), U.shape(0), U.shape(1));
  auto _J = J_t(J.data(), J.shape(0), J.shape(1));
  auto _K = K_t(K.data(), K.shape(0), K.shape(1));

  switch (map_type)
  {
  case maps::type::identity:
  {
    assert(_U.extent(0) == _u.extent(0));
    assert(_U.extent(1) == _u.extent(1));
    for (std::size_t i = 0; i < _U.extent(0); ++i)
      for (std::size_t j = 0; j < _U.extent(1); ++j)
        _u(i, j) = _U(i, j);
    return;
  }
  case maps::type::covariantPiola:
    maps::covariant_piola(_u, _U, _K, 1.0 / detJ, _J);
    return;
  case maps::type::contravariantPiola:
    maps::contravariant_piola(_u, _U, _K, 1.0 / detJ, _J);
    return;
  case maps::type::doubleCovariantPiola:
    maps::double_covariant_piola(_u, _U, _K, 1.0 / detJ, _J);
    return;
  case maps::type::doubleContravariantPiola:
    maps::double_contravariant_piola(_u, _U, _K, 1.0 / detJ, _J);
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
    auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
      return {pt[1], pt[0], 0.0};
    };
    const xt::xtensor<double, 2> J = {{0., 1.}, {1., 0.}};
    const double detJ = -1.;
    const xt::xtensor<double, 2> K = {{0., 1.}, {1., 0.}};
    mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    return mapinfo;
  }
  case cell::type::quadrilateral:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
      return {1 - pt[0], pt[1], 0};
    };
    const xt::xtensor<double, 2> J = {{-1., 0.}, {0., 1.}};
    const double detJ = -1.;
    const xt::xtensor<double, 2> K = {{-1., 0.}, {0., 1.}};
    mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    return mapinfo;
  }
  case cell::type::tetrahedron:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::interval] = {};
    mapinfo[cell::type::triangle] = {};
    {
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[0], pt[2], pt[1]};
      };
      const xt::xtensor<double, 2> J
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[2], pt[0], pt[1]};
      };
      const xt::xtensor<double, 2> J
          = {{0., 0., 1.}, {1., 0., 0.}, {0., 1., 0.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {0., 0., 1.}, {1., 0., 0.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[0], pt[2], pt[1]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[1], pt[0], pt[2]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[1], pt[0], pt[2]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[1] - pt[0], pt[0], pt[2]};
      };
      const xt::xtensor<double, 2> J
          = {{-1., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {-1., -1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[1], pt[0], pt[2]};
      };
      const xt::xtensor<double, 2> J
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[2], pt[1], pt[0]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[2], pt[1], pt[0]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      mapinfo[cell::type::interval].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[1], pt[0], pt[2]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[1], pt[0], pt[2]};
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
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[2] - pt[0], pt[1], pt[0]};
      };
      const xt::xtensor<double, 2> J
          = {{-1., 0., -1.}, {0., 1., 0.}, {1., 0., 0.}};
      const double detJ = 1.;
      const xt::xtensor<double, 2> K
          = {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., -1.}};
      mapinfo[cell::type::triangle].push_back(std::make_tuple(map, J, detJ, K));
    }
    { // scope
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[2], pt[1], pt[0]};
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
    const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const xt::xtensor<double, 2> J,
    const double detJ, const xt::xtensor<double, 2> K,
    const std::function<std::array<double, 3>(xtl::span<const double>)>
        map_point,
    int degree, int tdim, const int entity, std::size_t vs,
    const maps::type map_type)
{
  if (x[tdim].size() == 0 or x[tdim][entity].shape(0) == 0)
    return xt::xtensor<double, 2>({0, 0});

  const xt::xtensor<double, 2>& pts = x[tdim][entity];
  const xt::xtensor<double, 4>& imat = M[tdim][entity];

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
  {
    auto mp = map_point(xtl::span(pts.data() + p * pts.shape(1), pts.shape(1)));
    for (std::size_t k = 0; k < mapped_pts.shape(1); ++k)
      mapped_pts(p, k) = mp[k];
  }

  auto [polyset_vals_b, polyset_shape] = polyset::tabulate(
      cell_type, degree, 0,
      cmdspan2_t(mapped_pts.data(), mapped_pts.shape(0), mapped_pts.shape(1)));
  assert(polyset_shape[0] == 1);
  cmdspan2_t polyset_vals(polyset_vals_b.data(), polyset_shape[1],
                          polyset_shape[2]);

  xt::xtensor<double, 3> tabulated_data({npts, total_ndofs, vs});
  for (std::size_t j = 0; j < vs; ++j)
  {
    mdarray2_t result(polyset_vals.extent(1), coeffs.shape(0));
    for (std::size_t k0 = 0; k0 < coeffs.shape(0); ++k0)
      for (std::size_t k1 = 0; k1 < polyset_vals.extent(1); ++k1)
        for (std::size_t k2 = 0; k2 < polyset_vals.extent(0); ++k2)
          result(k1, k0) += coeffs(k0, k2 + psize * j) * polyset_vals(k2, k1);

    for (std::size_t k0 = 0; k0 < result.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < result.extent(1); ++k1)
        tabulated_data(k0, k1, j) = result(k0, k1);
  }

  // Pull back
  xt::xtensor<double, 3> pulled_data(tabulated_data.shape());
  xt::xtensor<double, 2> temp_data(
      {pulled_data.shape(1), pulled_data.shape(2)});
  for (std::size_t i = 0; i < npts; ++i)
  {
    pull_back(map_type, temp_data,
              xt::view(tabulated_data, i, xt::all(), xt::all()), J, detJ, K);
    for (std::size_t k0 = 0; k0 < temp_data.shape(0); ++k0)
      for (std::size_t k1 = 0; k1 < temp_data.shape(1); ++k1)
        pulled_data(i, k0, k1) = temp_data(k0, k1);
  }

  // Interpolate to calculate coefficients
  xt::xtensor<double, 2> transform = xt::zeros<double>({ndofs, ndofs});
  for (std::size_t d = 0; d < imat.shape(3); ++d)
  {
    for (std::size_t i = 0; i < vs; ++i)
    {
      for (std::size_t k0 = 0; k0 < transform.shape(1); ++k0)
        for (std::size_t k1 = 0; k1 < transform.shape(0); ++k1)
          for (std::size_t k2 = 0; k2 < imat.shape(2); ++k2)
            transform(k1, k0)
                += imat(k0, i, k2, d) * pulled_data(k2, k1 + dofstart, i);
    }
  }

  return transform;
}
//-----------------------------------------------------------------------------
std::map<cell::type, xt::xtensor<double, 3>>
compute_entity_transformations_impl(
    cell::type cell_type,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, int degree, std::size_t vs,
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
      xt::xtensor<double, 2> t2
          = compute_transformation(cell_type, x, M, coeffs, J, detJ, K, map,
                                   degree, tdim, entity, vs, map_type);

      for (std::size_t k0 = 0; k0 < transform.shape(1); ++k0)
        for (std::size_t k1 = 0; k1 < transform.shape(2); ++k1)
          transform(i, k0, k1) = t2(k0, k1);
    }

    out[item.first] = transform;
  }

  return out;
}
} // namespace
//-----------------------------------------------------------------------------
std::map<cell::type, std::pair<std::vector<double>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type cell_type,
    const std::array<
        std::vector<std::experimental::mdspan<
            const double, std::experimental::dextents<std::size_t, 2>>>,
        4>& x,
    const std::array<
        std::vector<std::experimental::mdspan<
            const double, std::experimental::dextents<std::size_t, 4>>>,
        4>& M,
    const std::experimental::mdspan<
        const double, std::experimental::dextents<std::size_t, 2>>& coeffs,
    int degree, std::size_t vs, maps::type map_type)
{
  auto out = compute_entity_transformations_impl(
      cell_type, to_xtensor(x), to_xtensor(M),
      xt::adapt(coeffs.data(), coeffs.size(), xt::no_ownership(),
                std::vector<std::size_t>{coeffs.extent(0), coeffs.extent(1)}),
      degree, vs, map_type);

  std::map<cell::type,
           std::pair<std::vector<double>, std::array<std::size_t, 3>>>
      trans;
  for (auto& data : out)
  {
    xt::xtensor<double, 3>& array = data.second;
    std::array<std::size_t, 3> s
        = {array.shape(0), array.shape(1), array.shape(2)};
    std::vector<double> a(array.data(), array.data() + array.size());
    trans.insert({data.first, {a, s}});
  }

  return trans;
}
//-----------------------------------------------------------------------------
