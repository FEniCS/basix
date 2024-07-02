// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-transformations.h"
#include "math.h"
#include "mdspan.hpp"
#include "polyset.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <span>
#include <tuple>

using namespace basix;

namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
template <typename T, std::size_t d>
using mdarray_t
    = stdex::mdarray<T,
                     MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

template <typename T>
using map_data_t
    = std::tuple<std::function<std::array<T, 3>(std::span<const T>)>,
                 mdarray_t<T, 2>, T, mdarray_t<T, 2>>;
template <typename T>
using mapinfo_t = std::map<cell::type, std::vector<map_data_t<T>>>;

namespace
{
//-----------------------------------------------------------------------------
int find_first_subentity(cell::type cell_type, cell::type entity_type)
{
  const int edim = cell::topological_dimension(entity_type);
  std::vector<cell::type> entities = cell::subentity_types(cell_type)[edim];
  if (auto it = std::ranges::find(entities, entity_type);
      it != entities.end())
  {
    return std::distance(entities.begin(), it);
  }
  else
    throw std::runtime_error("Entity not found");
}
//-----------------------------------------------------------------------------
template <typename Q, typename P, typename R, typename S>
void push_forward(maps::type map_type, Q&& u, const P& U, const R& J,
                  double detJ, const S& K)
{
  switch (map_type)
  {
  case maps::type::identity:
  {
    assert(U.extent(0) == u.extent(0));
    assert(U.extent(1) == u.extent(1));
    for (std::size_t i = 0; i < U.extent(0); ++i)
      for (std::size_t j = 0; j < U.extent(1); ++j)
        u[i, j] = U[i, j];
    return;
  }
  case maps::type::covariantPiola:
    maps::covariant_piola(u, U, J, detJ, K);
    return;
  case maps::type::contravariantPiola:
    maps::contravariant_piola(u, U, J, detJ, K);
    return;
  case maps::type::doubleCovariantPiola:
    maps::double_covariant_piola(u, U, J, detJ, K);
    return;
  case maps::type::doubleContravariantPiola:
    maps::double_contravariant_piola(u, U, J, detJ, K);
    return;
  default:
    throw std::runtime_error("Map not implemented");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
mapinfo_t<T> get_mapinfo(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::point:
    return mapinfo_t<T>();
  case cell::type::interval:
    return mapinfo_t<T>();
  case cell::type::triangle:
  {
    mapinfo_t<T> mapinfo;
    auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
    auto map = [](auto pt) -> std::array<T, 3> {
      return {pt[1], pt[0], 0.0};
    };
    stdex::mdarray<T, stdex::extents<std::size_t, 2, 2>> J(
        stdex::extents<std::size_t, 2, 2>{}, {0., 1., 1., 0.});

    T detJ = -1.0;
    stdex::mdarray<T, stdex::extents<std::size_t, 2, 2>> K(
        stdex::extents<std::size_t, 2, 2>{}, {0., 1., 1., 0.});
    data.push_back(std::tuple(map, J, detJ, K));
    return mapinfo;
  }
  case cell::type::quadrilateral:
  {
    mapinfo_t<T> mapinfo;
    auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
    auto map = [](auto pt) -> std::array<T, 3> {
      return {1 - pt[0], pt[1], 0};
    };
    stdex::mdarray<T, stdex::extents<std::size_t, 2, 2>> J(
        stdex::extents<std::size_t, 2, 2>{}, {-1., 0., 0., 1.});

    T detJ = -1.0;
    stdex::mdarray<T, stdex::extents<std::size_t, 2, 2>> K(
        stdex::extents<std::size_t, 2, 2>{}, {-1., 0., 0., 1.});
    data.push_back(std::tuple(map, J, detJ, K));
    return mapinfo;
  }
  case cell::type::tetrahedron:
  {
    mapinfo_t<T> mapinfo;
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](auto pt) -> std::array<T, 3> {
        return {pt[0], pt[2], pt[1]};
      };
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
          stdex::extents<std::size_t, 3, 3>{},
          {1., 0., 0., 0., 0., 1., 0., 1., 0.});

      T detJ = -1.0;
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
          stdex::extents<std::size_t, 3, 3>{},
          {1., 0., 0., 0., 0., 1., 0., 1., 0.});
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[2], pt[0], pt[1]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 0., 0., 1., 1., 0., 0.});

        T detJ = 1.0;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 1., 0., 0., 0., 1., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[0], pt[2], pt[1]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {1., 0., 0., 0., 0., 1., 0., 1., 0.});

        T detJ = -1.0;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {1., 0., 0., 0., 0., 1., 0., 1., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    return mapinfo;
  }
  case cell::type::hexahedron:
  {
    mapinfo_t<T> mapinfo;
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](auto pt) -> std::array<T, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
          stdex::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});

      T detJ = -1.0;
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
          stdex::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {1 - pt[1], pt[0], pt[2]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., -1., 0., 0., 0., 0., 1.});

        T detJ = 1.0;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., -1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[1], pt[0], pt[2]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});

        T detJ = -1.0;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }
    return mapinfo;
  }
  case cell::type::prism:
  {
    mapinfo_t<T> mapinfo;
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](auto pt) -> std::array<T, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
          stdex::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      T detJ = -1.0;
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
          stdex::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      data.push_back(std::tuple(map, J, detJ, K));
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {1 - pt[1] - pt[0], pt[0], pt[2]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., -1., -1., 0., 0., 0., 1.});
        T detJ = 1.0;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {-1., -1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[1], pt[0], pt[2]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        T detJ = -1.;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {1 - pt[2], pt[1], pt[0]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., -1., 0., 0.});
        T detJ = 1.0;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., -1., 0., 1., 0., 1., 0., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      { // scope
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[2], pt[1], pt[0]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., 1., 0., 0.});
        T detJ = -1.;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., 1., 0., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    return mapinfo;
  }
  case cell::type::pyramid:
  {
    mapinfo_t<T> mapinfo;
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map = [](auto pt) -> std::array<T, 3> {
        return {1 - pt[0], pt[1], pt[2]};
      };
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
          stdex::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      T detJ = -1.;
      stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
          stdex::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {1 - pt[1], pt[0], pt[2]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., -1., 0., 0., 0., 0., 1.});
        T detJ = 1.;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., -1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[1], pt[0], pt[2]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        T detJ = -1.;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {1 - pt[2] - pt[0], pt[1], pt[0]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., -1., 0., -1.});
        T detJ = 1.;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {-1., 0., -1., 0., 1., 0., 1., 0., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map = [](auto pt) -> std::array<T, 3> {
          return {pt[2], pt[1], pt[0]};
        };
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> J(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., 1., 0., 0.});
        T detJ = -1.;
        stdex::mdarray<T, stdex::extents<std::size_t, 3, 3>> K(
            stdex::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., 1., 0., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    return mapinfo;
  }
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>> compute_transformation(
    cell::type cell_type,
    const std::array<std::vector<mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<mdspan_t<const T, 4>>, 4>& M,
    mdspan_t<const T, 2> coeffs, const mdarray_t<T, 2>& J, T detJ,
    const mdarray_t<T, 2>& K,
    const std::function<std::array<T, 3>(std::span<const T>)> map_point,
    int degree, int tdim, int entity, std::size_t vs, const maps::type map_type,
    const polyset::type ptype)
{
  if (x[tdim].size() == 0 or x[tdim][entity].extent(0) == 0)
    return {{}, {0, 0}};

  mdspan_t<const T, 2> pts = x[tdim][entity];
  mdspan_t<const T, 4> imat = M[tdim][entity];

  const std::size_t ndofs = imat.extent(0);
  const std::size_t npts = pts.extent(0);
  const int psize = polyset::dim(cell_type, ptype, degree);

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
  mdarray_t<T, 2> mapped_pts(pts.extents());
  for (std::size_t p = 0; p < mapped_pts.extent(0); ++p)
  {
    auto mp = map_point(
        std::span(pts.data_handle() + p * pts.extent(1), pts.extent(1)));
    for (std::size_t k = 0; k < mapped_pts.extent(1); ++k)
      mapped_pts[p, k] = mp[k];
  }

  auto [polyset_vals_b, polyset_shape] = polyset::tabulate(
      cell_type, ptype, degree, 0,
      mdspan_t<const T, 2>(mapped_pts.data(), mapped_pts.extents()));
  assert(polyset_shape[0] == 1);
  mdspan_t<const T, 2> polyset_vals(polyset_vals_b.data(), polyset_shape[1],
                                    polyset_shape[2]);

  mdarray_t<T, 3> tabulated_data(npts, total_ndofs, vs);
  for (std::size_t j = 0; j < vs; ++j)
  {
    mdarray_t<T, 2> result(polyset_vals.extent(1), coeffs.extent(0));
    for (std::size_t k0 = 0; k0 < coeffs.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < polyset_vals.extent(1); ++k1)
        for (std::size_t k2 = 0; k2 < polyset_vals.extent(0); ++k2)
          result[k1, k0] += coeffs[k0, k2 + psize * j] * polyset_vals[k2, k1];

    for (std::size_t k0 = 0; k0 < result.extent(0); ++k0)
      for (std::size_t k1 = 0; k1 < result.extent(1); ++k1)
        tabulated_data[k0, k1, j] = result[k0, k1];
  }

  // push forward
  mdarray_t<T, 3> pushed_data(tabulated_data.extents());
  {
    mdarray_t<T, 2> temp_data(pushed_data.extent(1), pushed_data.extent(2));
    for (std::size_t i = 0; i < npts; ++i)
    {
      mdspan_t<const T, 2> tab(
          tabulated_data.data()
              + i * tabulated_data.extent(1) * tabulated_data.extent(2),
          tabulated_data.extent(1), tabulated_data.extent(2));

      push_forward(map_type,
                   mdspan_t<T, 2>(temp_data.data(), temp_data.extents()), tab,
                   J, detJ, K);

      for (std::size_t k0 = 0; k0 < temp_data.extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < temp_data.extent(1); ++k1)
          pushed_data[i, k0, k1] = temp_data[k0, k1];
    }
  }

  // Interpolate to calculate coefficients
  std::vector<T> transformb(ndofs * ndofs);
  mdspan_t<T, 2> transform(transformb.data(), ndofs, ndofs);
  for (std::size_t d = 0; d < imat.extent(3); ++d)
  {
    for (std::size_t i = 0; i < vs; ++i)
    {
      for (std::size_t k0 = 0; k0 < transform.extent(1); ++k0)
        for (std::size_t k1 = 0; k1 < transform.extent(0); ++k1)
          for (std::size_t k2 = 0; k2 < imat.extent(2); ++k2)
            transform[k1, k0]
                += imat[k0, i, k2, d] * pushed_data[k2, k1 + dofstart, i];
    }
  }

  return {std::move(transformb), {transform.extent(0), transform.extent(1)}};
}
} // namespace
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::map<cell::type, std::pair<std::vector<T>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type cell_type,
    const std::array<
        std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>>,
        4>& x,
    const std::array<
        std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>>,
        4>& M,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        coeffs,
    int degree, std::size_t vs, maps::type map_type, polyset::type ptype)
{
  std::map<cell::type, std::pair<std::vector<T>, std::array<std::size_t, 3>>>
      out;
  const mapinfo_t<T> mapinfo = get_mapinfo<T>(cell_type);
  for (auto& [entity_type, emap_data] : mapinfo)
  {
    const int tdim = cell::topological_dimension(entity_type);
    const int entity = find_first_subentity(cell_type, entity_type);
    std::size_t ndofs = M[tdim].size() == 0 ? 0 : M[tdim][entity].extent(0);

    std::vector<T> transform;
    transform.reserve(emap_data.size() * ndofs * ndofs);
    for (auto& [mapfn, J, detJ, K] : emap_data)
    {
      auto [t2b, _]
          = compute_transformation(cell_type, x, M, coeffs, J, detJ, K, mapfn,
                                   degree, tdim, entity, vs, map_type, ptype);
      transform.insert(transform.end(), t2b.begin(), t2b.end());
    }

    out.try_emplace(entity_type,
                    std::pair(std::move(transform),
                              std::array{emap_data.size(), ndofs, ndofs}));
  }

  return out;
}
//-----------------------------------------------------------------------------
/// @cond
template std::map<cell::type,
                  std::pair<std::vector<float>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type,
    const std::array<std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                         const float, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<
                                          std::size_t, 2>>>,
                     4>&,
    const std::array<std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                         const float, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<
                                          std::size_t, 4>>>,
                     4>&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const float, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int, std::size_t, maps::type, polyset::type);

template std::map<cell::type,
                  std::pair<std::vector<double>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type,
    const std::array<std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                         const double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<
                                           std::size_t, 2>>>,
                     4>&,
    const std::array<std::vector<MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                         const double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<
                                           std::size_t, 4>>>,
                     4>&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int, std::size_t, maps::type, polyset::type);
/// @endcond
//-----------------------------------------------------------------------------
