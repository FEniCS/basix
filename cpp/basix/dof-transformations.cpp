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
using mdarray3_t = stdex::mdarray<double, stdex::dextents<std::size_t, 3>>;

using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;

using map_data_t
    = std::tuple<std::function<std::array<double, 3>(xtl::span<const double>)>,
                 xt::xtensor<double, 2>, double, xt::xtensor<double, 2>>;
typedef std::map<cell::type, std::vector<map_data_t>> mapinfo_t;

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
void pull_back(maps::type map_type, mdspan2_t u, cmdspan2_t U, cmdspan2_t J,
               double detJ, cmdspan2_t K)
{
  switch (map_type)
  {
  case maps::type::identity:
  {
    assert(U.extent(0) == u.extent(0));
    assert(U.extent(1) == u.extent(1));
    for (std::size_t i = 0; i < U.extent(0); ++i)
      for (std::size_t j = 0; j < U.extent(1); ++j)
        u(i, j) = U(i, j);
    return;
  }
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
    auto& data = mapinfo.try_emplace(cell::type::interval).first->second;

    auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
      return {pt[1], pt[0], 0.0};
    };
    const xt::xtensor<double, 2> J = {{0., 1.}, {1., 0.}};
    const double detJ = -1.;
    const xt::xtensor<double, 2> K = {{0., 1.}, {1., 0.}};
    data.push_back(std::tuple(map, J, detJ, K));

    return mapinfo;
  }
  case cell::type::quadrilateral:
  {
    mapinfo_t mapinfo;
    auto& data = mapinfo.try_emplace(cell::type::interval).first->second;

    auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
      return {1 - pt[0], pt[1], 0};
    };
    const xt::xtensor<double, 2> J = {{-1., 0.}, {0., 1.}};
    const double detJ = -1.;
    const xt::xtensor<double, 2> K = {{-1., 0.}, {0., 1.}};
    data.push_back(std::tuple(map, J, detJ, K));

    return mapinfo;
  }
  case cell::type::tetrahedron:
  {
    mapinfo_t mapinfo;
    // mapinfo[cell::type::interval] = {};
    mapinfo[cell::type::triangle] = {};
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;

      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {pt[0], pt[2], pt[1]};
      };
      const xt::xtensor<double, 2> J
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      const double detJ = -1.;
      const xt::xtensor<double, 2> K
          = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[2], pt[0], pt[1]};
        };
        xt::xtensor<double, 2> J = {{0., 0., 1.}, {1., 0., 0.}, {0., 1., 0.}};
        double detJ = 1.0;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {0., 0., 1.}, {1., 0., 0.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[0], pt[2], pt[1]};
        };
        xt::xtensor<double, 2> J = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
        double detJ = -1.0;
        xt::xtensor<double, 2> K = {{1., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    return mapinfo;
  }
  case cell::type::hexahedron:
  {
    mapinfo_t mapinfo;
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      xt::xtensor<double, 2> J = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      double detJ = -1.9;
      xt::xtensor<double, 2> K = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      data.push_back(std::tuple(map, J, detJ, K));
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {1 - pt[1], pt[0], pt[2]};
        };
        xt::xtensor<double, 2> J = {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        double detJ = 1.0;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {-1., 0., 0.}, {0., 0., 1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[1], pt[0], pt[2]};
        };
        xt::xtensor<double, 2> J = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        double detJ = -1.0;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }
    return mapinfo;
  }
  case cell::type::prism:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::quadrilateral] = {};
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      xt::xtensor<double, 2> J = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      double detJ = -1.0;
      xt::xtensor<double, 2> K = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      data.push_back(std::tuple(map, J, detJ, K));
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {1 - pt[1] - pt[0], pt[0], pt[2]};
        };
        xt::xtensor<double, 2> J = {{-1., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        double detJ = 1.0;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {-1., -1., 0.}, {0., 0., 1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[1], pt[0], pt[2]};
        };
        xt::xtensor<double, 2> J = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        double detJ = -1.;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {1 - pt[2], pt[1], pt[0]};
        };
        xt::xtensor<double, 2> J = {{0., 0., -1.}, {0., 1., 0.}, {1., 0., 0.}};
        double detJ = 1.0;
        xt::xtensor<double, 2> K = {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
      { // scope
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[2], pt[1], pt[0]};
        };
        xt::xtensor<double, 2> J = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
        double detJ = -1.;
        xt::xtensor<double, 2> K = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    return mapinfo;
  }
  case cell::type::pyramid:
  {
    mapinfo_t mapinfo;
    mapinfo[cell::type::triangle] = {};
    mapinfo[cell::type::quadrilateral] = {};
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      xt::xtensor<double, 2> J = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      double detJ = -1.;
      xt::xtensor<double, 2> K = {{-1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {1 - pt[1], pt[0], pt[2]};
        };
        xt::xtensor<double, 2> J = {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        double detJ = 1.;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {-1., 0., 0.}, {0., 0., 1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[1], pt[0], pt[2]};
        };
        xt::xtensor<double, 2> J = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        double detJ = -1.;
        xt::xtensor<double, 2> K = {{0., 1., 0.}, {1., 0., 0.}, {0., 0., 1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {1 - pt[2] - pt[0], pt[1], pt[0]};
        };
        xt::xtensor<double, 2> J = {{-1., 0., -1.}, {0., 1., 0.}, {1., 0., 0.}};
        double detJ = 1.;
        xt::xtensor<double, 2> K = {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., -1.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](xtl::span<const double> pt) -> std::array<double, 3> {
          return {pt[2], pt[1], pt[0]};
        };
        xt::xtensor<double, 2> J = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
        double detJ = -1.;
        xt::xtensor<double, 2> K = {{0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    return mapinfo;
  }
  default:
    throw std::runtime_error("Unsupported cell type");
  } // namespace
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
compute_transformation(
    cell::type cell_type, const std::array<std::vector<cmdspan2_t>, 4>& x,
    const std::array<std::vector<cmdspan4_t>, 4>& M, cmdspan2_t coeffs,
    cmdspan2_t J, double detJ, cmdspan2_t K,
    const std::function<std::array<double, 3>(xtl::span<const double>)>
        map_point,
    int degree, int tdim, const int entity, std::size_t vs,
    const maps::type map_type)
{
  if (x[tdim].size() == 0 or x[tdim][entity].extent(0) == 0)
    return {{}, {0, 0}};

  cmdspan2_t pts = x[tdim][entity];
  cmdspan4_t imat = M[tdim][entity];

  const std::size_t ndofs = imat.extent(0);
  const std::size_t npts = pts.extent(0);
  const int psize = polyset::dim(cell_type, degree);

  std::size_t dofstart = 0;
  for (int d = 0; d < tdim; ++d)
    for (std::size_t i = 0; i < M[d].size(); ++i)
      dofstart += M[d][i].extent(0);
  for (int i = 0; i < entity; ++i)
    dofstart += M[tdim][i].extent(0);

  std::size_t total_ndofs = 0;
  for (int d = 0; d <= 3; ++d)
    for (std::size_t i = 0; i < M[d].size(); ++i)
      total_ndofs += M[d][i].extent(0);

  // Map the points to reverse the edge, then tabulate at those points
  mdarray2_t mapped_pts(pts.extents());
  for (std::size_t p = 0; p < mapped_pts.extent(0); ++p)
  {
    auto mp
        = map_point(xtl::span(pts.data() + p * pts.extent(1), pts.extent(1)));
    for (std::size_t k = 0; k < mapped_pts.extent(1); ++k)
      mapped_pts(p, k) = mp[k];
  }

  auto [polyset_vals_b, polyset_shape]
      = polyset::tabulate(cell_type, degree, 0,
                          cmdspan2_t(mapped_pts.data(), mapped_pts.extents()));
  assert(polyset_shape[0] == 1);
  cmdspan2_t polyset_vals(polyset_vals_b.data(), polyset_shape[1],
                          polyset_shape[2]);

  mdarray3_t tabulated_data(npts, total_ndofs, vs);
  for (std::size_t j = 0; j < vs; ++j)
  {
    mdarray2_t result(polyset_vals.extent(1), coeffs.extent(0));
    for (std::size_t k0 = 0; k0 < coeffs.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < polyset_vals.extent(1); ++k1)
        for (std::size_t k2 = 0; k2 < polyset_vals.extent(0); ++k2)
          result(k1, k0) += coeffs(k0, k2 + psize * j) * polyset_vals(k2, k1);

    for (std::size_t k0 = 0; k0 < result.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < result.extent(1); ++k1)
        tabulated_data(k0, k1, j) = result(k0, k1);
  }

  // Pull back
  mdarray3_t pulled_data(tabulated_data.extents());
  {
    std::vector<double> temp_data_b(pulled_data.extent(1)
                                    * pulled_data.extent(2));
    mdspan2_t temp_data(temp_data_b.data(), pulled_data.extent(1),
                        pulled_data.extent(2));
    for (std::size_t i = 0; i < npts; ++i)
    {
      cmdspan2_t tab(tabulated_data.data()
                         + i * tabulated_data.extent(1)
                               * tabulated_data.extent(2),
                     tabulated_data.extent(1), tabulated_data.extent(2));

      pull_back(map_type, temp_data, tab, J, detJ, K);

      for (std::size_t k0 = 0; k0 < temp_data.extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < temp_data.extent(1); ++k1)
          pulled_data(i, k0, k1) = temp_data(k0, k1);
    }
  }

  // Interpolate to calculate coefficients
  std::vector<double> transformb(ndofs * ndofs);
  mdspan2_t transform(transformb.data(), ndofs, ndofs);
  for (std::size_t d = 0; d < imat.extent(3); ++d)
  {
    for (std::size_t i = 0; i < vs; ++i)
    {
      for (std::size_t k0 = 0; k0 < transform.extent(1); ++k0)
        for (std::size_t k1 = 0; k1 < transform.extent(0); ++k1)
          for (std::size_t k2 = 0; k2 < imat.extent(2); ++k2)
            transform(k1, k0)
                += imat(k0, i, k2, d) * pulled_data(k2, k1 + dofstart, i);
    }
  }

  return {std::move(transformb), {transform.extent(0), transform.extent(1)}};
}
//-----------------------------------------------------------------------------
std::map<cell::type, xt::xtensor<double, 3>>
compute_entity_transformations_impl(
    cell::type cell_type, const std::array<std::vector<cmdspan2_t>, 4>& x,
    const std::array<std::vector<cmdspan4_t>, 4>& M, cmdspan2_t coeffs,
    int degree, std::size_t vs, maps::type map_type)
{
  std::map<cell::type, xt::xtensor<double, 3>> out;

  mapinfo_t mapinfo = get_mapinfo(cell_type);
  for (auto item : mapinfo)
  {
    const int tdim = cell::topological_dimension(item.first);
    const int entity = find_first_subentity(cell_type, item.first);
    const std::size_t ndofs
        = M[tdim].size() == 0 ? 0 : M[tdim][entity].extent(0);
    xt::xtensor<double, 3> transform({item.second.size(), ndofs, ndofs});
    for (std::size_t i = 0; i < item.second.size(); ++i)
    {
      auto [map, J, detJ, K] = item.second[i];
      const auto [t2b, tshape] = compute_transformation(
          cell_type, x, M, coeffs, cmdspan2_t(J.data(), J.shape(0), J.shape(1)),
          detJ, cmdspan2_t(K.data(), K.shape(0), K.shape(1)), map, degree, tdim,
          entity, vs, map_type);
      cmdspan2_t t2(t2b.data(), tshape);
      for (std::size_t k0 = 0; k0 < transform.shape(1); ++k0)
        for (std::size_t k1 = 0; k1 < transform.shape(2); ++k1)
          transform(i, k0, k1) = t2(k0, k1);
    }

    out.insert({item.first, transform});
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
  std::map<cell::type, xt::xtensor<double, 3>> out
      = compute_entity_transformations_impl(cell_type, x, M, coeffs, degree, vs,
                                            map_type);

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
