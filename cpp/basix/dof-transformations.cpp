// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
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

namespace
{
template <typename T, std::size_t d>
using mdarray_t = mdex::mdarray<T, md::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdspan_t = md::mdspan<T, md::dextents<std::size_t, d>>;

template <typename T>
using map_data_t
    = std::tuple<std::function<std::array<T, 3>(std::span<const T>)>,
                 mdarray_t<T, 2>, T, mdarray_t<T, 2>>;
template <typename T>
using mapinfo_t = std::map<cell::type, std::vector<map_data_t<T>>>;

//-----------------------------------------------------------------------------
int find_first_subentity(cell::type cell_type, cell::type entity_type)
{
  const int edim = cell::topological_dimension(entity_type);
  std::vector<cell::type> entities = cell::subentity_types(cell_type)[edim];
  if (auto it = std::ranges::find(entities, entity_type); it != entities.end())
    return std::distance(entities.begin(), it);
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
        u(i, j) = U(i, j);
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
    auto map = [](auto pt) -> std::array<T, 3> { return {pt[1], pt[0], 0}; };
    mdex::mdarray<T, md::extents<std::size_t, 2, 2>> J(
        md::extents<std::size_t, 2, 2>{}, {0., 1., 1., 0.});

    T detJ = -1;
    mdex::mdarray<T, md::extents<std::size_t, 2, 2>> K(
        md::extents<std::size_t, 2, 2>{}, {0., 1., 1., 0.});
    data.push_back(std::tuple(map, J, detJ, K));
    return mapinfo;
  }
  case cell::type::quadrilateral:
  {
    mapinfo_t<T> mapinfo;
    auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
    auto map
        = [](auto pt) -> std::array<T, 3> { return {1 - pt[0], pt[1], 0}; };
    mdex::mdarray<T, md::extents<std::size_t, 2, 2>> J(
        md::extents<std::size_t, 2, 2>{}, {-1., 0., 0., 1.});

    T detJ = -1.0;
    mdex::mdarray<T, md::extents<std::size_t, 2, 2>> K(
        md::extents<std::size_t, 2, 2>{}, {-1., 0., 0., 1.});
    data.push_back(std::tuple(map, J, detJ, K));
    return mapinfo;
  }
  case cell::type::tetrahedron:
  {
    mapinfo_t<T> mapinfo;
    {
      auto& data = mapinfo.try_emplace(cell::type::interval).first->second;
      auto map
          = [](auto pt) -> std::array<T, 3> { return {pt[0], pt[2], pt[1]}; };
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
          md::extents<std::size_t, 3, 3>{},
          {1., 0., 0., 0., 0., 1., 0., 1., 0.});

      T detJ = -1.0;
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
          md::extents<std::size_t, 3, 3>{},
          {1., 0., 0., 0., 0., 1., 0., 1., 0.});
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[2], pt[0], pt[1]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 0., 0., 1., 1., 0., 0.});

        T detJ = 1.0;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 1., 0., 0., 0., 1., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[0], pt[2], pt[1]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {1., 0., 0., 0., 0., 1., 0., 1., 0.});

        T detJ = -1.0;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
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
      auto map = [](auto pt) -> std::array<T, 3>
      { return {1 - pt[0], pt[1], pt[2]}; };
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
          md::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});

      T detJ = -1.0;
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
          md::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3>
        { return {1 - pt[1], pt[0], pt[2]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., -1., 0., 0., 0., 0., 1.});

        T detJ = 1.0;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {0., -1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[1], pt[0], pt[2]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});

        T detJ = -1.0;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
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
      auto map = [](auto pt) -> std::array<T, 3>
      { return {1 - pt[0], pt[1], pt[2]}; };
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
          md::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      T detJ = -1.0;
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
          md::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      data.push_back(std::tuple(map, J, detJ, K));
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3>
        { return {1 - pt[1] - pt[0], pt[0], pt[2]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., -1., -1., 0., 0., 0., 1.});
        T detJ = 1.0;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {-1., -1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[1], pt[0], pt[2]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        T detJ = -1.;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }
    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3>
        { return {1 - pt[2], pt[1], pt[0]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., -1., 0., 0.});
        T detJ = 1.0;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {0., 0., -1., 0., 1., 0., 1., 0., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      { // scope
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[2], pt[1], pt[0]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., 1., 0., 0.});
        T detJ = -1.;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
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
      auto map = [](auto pt) -> std::array<T, 3>
      { return {1 - pt[0], pt[1], pt[2]}; };
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
          md::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      T detJ = -1.;
      mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
          md::extents<std::size_t, 3, 3>{},
          {-1., 0., 0., 0., 1., 0., 0., 0., 1.});
      data.push_back(std::tuple(map, J, detJ, K));
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::quadrilateral).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3>
        { return {1 - pt[1], pt[0], pt[2]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., -1., 0., 0., 0., 0., 1.});
        T detJ = 1.;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {0., -1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[1], pt[0], pt[2]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        T detJ = -1.;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {0., 1., 0., 1., 0., 0., 0., 0., 1.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
    }

    {
      auto& data = mapinfo.try_emplace(cell::type::triangle).first->second;
      {
        auto map = [](auto pt) -> std::array<T, 3>
        { return {1 - pt[2] - pt[0], pt[1], pt[0]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., -1., 0., -1.});
        T detJ = 1.;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
            {-1., 0., -1., 0., 1., 0., 1., 0., 0.});
        data.push_back(std::tuple(map, J, detJ, K));
      }
      {
        auto map
            = [](auto pt) -> std::array<T, 3> { return {pt[2], pt[1], pt[0]}; };
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> J(
            md::extents<std::size_t, 3, 3>{},
            {0., 0., 1., 0., 1., 0., 1., 0., 0.});
        T detJ = -1.;
        mdex::mdarray<T, md::extents<std::size_t, 3, 3>> K(
            md::extents<std::size_t, 3, 3>{},
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
    cell::type cell_type, std::array<std::vector<mdspan_t<const T, 2>>, 4> x,
    std::array<std::vector<mdspan_t<const T, 4>>, 4> M,
    mdspan_t<const T, 2> coeffs, const mdarray_t<T, 2>& J, T detJ,
    const mdarray_t<T, 2>& K,
    std::function<std::array<T, 3>(std::span<const T>)> map_point, int degree,
    int tdim, int entity, std::size_t vs, const maps::type map_type,
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
    std::array<T, 3> mp = map_point(
        std::span(pts.data_handle() + p * pts.extent(1), pts.extent(1)));
    for (std::size_t k = 0; k < mapped_pts.extent(1); ++k)
      mapped_pts(p, k) = mp[k];
  }

  auto [polyset_vals_b, polyset_shape] = polyset::tabulate(
      cell_type, ptype, degree, 0,
      mdspan_t<const T, 2>(mapped_pts.data(), mapped_pts.extents()));
  assert(polyset_shape[0] == 1);
  mdspan_t<const T, 2> polyset_vals(polyset_vals_b.data(), polyset_shape[1],
                                    polyset_shape[2]);

  std::vector<T> tabulated_data_b(npts * total_ndofs * vs);
  mdspan_t<T, 3> tabulated_data(tabulated_data_b.data(), npts, total_ndofs, vs);

  // coeffs has shape (total_ndofs, vs * psize) in row-major storage, i.e.
  // each dof's row is vs contiguous blocks of psize values, one per
  // value component. So viewing its buffer directly with shape
  // (total_ndofs * vs, psize) is exact (no data movement) and lets every
  // value component be computed by a single math::dot call, instead of
  // vs separate calls each preceded by an O(total_ndofs * psize) copy to
  // extract that component's columns out of coeffs -- the same pattern
  // fixed in FiniteElement::tabulate().
  assert(coeffs.extent(1) == vs * static_cast<std::size_t>(psize));
  mdspan_t<const T, 2> coeffs_flat(coeffs.data_handle(),
                                   coeffs.extent(0) * vs, psize);
  std::vector<T> result_b(polyset_vals.extent(1) * coeffs.extent(0) * vs);
  mdspan_t<T, 2> result(result_b.data(), coeffs.extent(0) * vs,
                        polyset_vals.extent(1));
  math::dot(coeffs_flat, polyset_vals, result);
  for (std::size_t k0 = 0; k0 < result.extent(1); ++k0)
    for (std::size_t k1 = 0; k1 < coeffs.extent(0); ++k1)
      for (std::size_t j = 0; j < vs; ++j)
        tabulated_data(k0, k1, j) = result(k1 * vs + j, k0);

  // Push forward.
  //
  // J, K and detJ are the same for every point here (this maps the
  // reference cell to itself, e.g. a reflection/rotation, not physical
  // cell geometry that could vary per point), so instead of calling
  // push_forward once per point, every point's (total_ndofs, vs) block
  // is pushed forward in a single call by viewing the whole
  // (npts, total_ndofs, vs) buffer as one flat (npts * total_ndofs, vs)
  // matrix -- valid since each row of that matrix is transformed
  // independently of the others.
  mdarray_t<T, 3> pushed_data(tabulated_data.extents());
  {
    mdspan_t<const T, 2> tab_all(tabulated_data_b.data(),
                                 npts * tabulated_data.extent(1),
                                 tabulated_data.extent(2));
    mdspan_t<T, 2> pushed_all(pushed_data.data(),
                              npts * pushed_data.extent(1),
                              pushed_data.extent(2));
    push_forward(map_type, pushed_all, tab_all, J, detJ, K);
  }

  // Interpolate to calculate coefficients
  std::vector<T> transformb(ndofs * ndofs);
  mdspan_t<T, 2> transform(transformb.data(), ndofs, ndofs);

  std::vector<T> imat_b(transform.extent(1) * imat.extent(2));
  mdspan_t<T, 2> imat_id(imat_b.data(), transform.extent(1), imat.extent(2));

  std::vector<T> pushed_datai_b(imat.extent(2) * transform.extent(0));
  mdspan_t<T, 2> pushed_data_i(pushed_datai_b.data(), imat.extent(2),
                               transform.extent(0));

  std::vector<T> transformT_b(transform.extent(1) * transform.extent(0));
  mdspan_t<T, 2> transformT(transformT_b.data(), transform.extent(1),
                            transform.extent(0));
  for (std::size_t i = 0; i < vs; ++i)
  {
    // Pack pushed_data
    for (std::size_t k2 = 0; k2 < imat.extent(2); ++k2)
      for (std::size_t k1 = 0; k1 < transform.extent(0); ++k1)
        pushed_data_i(k2, k1) = pushed_data(k2, k1 + dofstart, i);

    for (std::size_t d = 0; d < imat.extent(3); ++d)
    {
      // Pack imat
      for (std::size_t k0 = 0; k0 < transform.extent(1); ++k0)
        for (std::size_t k2 = 0; k2 < imat.extent(2); ++k2)
          imat_id(k0, k2) = imat(k0, i, k2, d);

      // transformT_(k0, k1) = imat_id_(k0, k2) pushed_data_i(k2, k1) +
      // transformT_(k0, k1)
      math::dot(imat_id, pushed_data_i, transformT, 1, 1);
    }
  }

  // Transpose 'transformT' -> 'transform'
  for (std::size_t k0 = 0; k0 < transform.extent(1); ++k0)
    for (std::size_t k1 = 0; k1 < transform.extent(0); ++k1)
      transform(k1, k0) = transformT(k0, k1);

  return {std::move(transformb), {transform.extent(0), transform.extent(1)}};
}
} // namespace
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::map<cell::type, std::pair<std::vector<T>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type cell_type,
    std::array<std::vector<md::mdspan<const T, md::dextents<std::size_t, 2>>>,
               4>
        x,
    std::array<std::vector<md::mdspan<const T, md::dextents<std::size_t, 4>>>,
               4>
        M,
    md::mdspan<const T, md::dextents<std::size_t, 2>> coeffs, int degree,
    std::size_t vs, maps::type map_type, polyset::type ptype)
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
/// @cond DOXYGEN_SHOULD_SKIP_THIS
template std::map<cell::type,
                  std::pair<std::vector<float>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type,
    std::array<
        std::vector<md::mdspan<const float, md::dextents<std::size_t, 2>>>, 4>,
    std::array<
        std::vector<md::mdspan<const float, md::dextents<std::size_t, 4>>>, 4>,
    md::mdspan<const float, md::dextents<std::size_t, 2>>, int, std::size_t,
    maps::type, polyset::type);

template std::map<cell::type,
                  std::pair<std::vector<double>, std::array<std::size_t, 3>>>
doftransforms::compute_entity_transformations(
    cell::type,
    std::array<
        std::vector<md::mdspan<const double, md::dextents<std::size_t, 2>>>, 4>,
    std::array<
        std::vector<md::mdspan<const double, md::dextents<std::size_t, 4>>>, 4>,
    md::mdspan<const double, md::dextents<std::size_t, 2>>, int, std::size_t,
    maps::type, polyset::type);
/// @endcond
//-----------------------------------------------------------------------------
