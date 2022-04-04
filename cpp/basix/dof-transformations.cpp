// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-transformations.h"
#include "math.h"
#include "polyset.h"
#include <algorithm>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
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
xt::xtensor<double, 2> compute_transformation(
    cell::type cell_type,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const xt::xtensor<double, 2> J,
    const double detJ, const xt::xtensor<double, 2> K,
    const std::function<xt::xtensor<double, 1>(const xt::xtensor<double, 1>&)>
        map_point,
    const int degree, const int tdim, const int vs, const maps::type map_type)
{
  if (x[tdim].size() == 0 or x[tdim][0].shape(0) == 0)
    return xt::xtensor<double, 2>({0, 0});

  const xt::xtensor<double, 2>& pts = x[1][0];
  const xt::xtensor<double, 3>& imat = M[1][0];

  const std::size_t ndofs = imat.shape(0);
  const std::size_t npts = pts.shape(0);
  const int psize = polyset::dim(cell_type, degree);

  std::size_t dofstart = 0;
  for (int d = 0; d < tdim; ++d)
    for (std::size_t i = 0; i < M[0].size(); ++i)
      dofstart += M[0][i].shape(0);

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
    auto result = math::dot(polyset_vals, xt::transpose(C));
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

    transform += math::dot(mat, values);
  }
  return transform;
}
//-----------------------------------------------------------------------------
std::map<cell::type, xt::xtensor<double, 3>>
compute_entity_transformations_triangle(
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const int degree, const int vs,
    maps::type map_type)
{
  std::map<cell::type, xt::xtensor<double, 3>> out;
  if (x[1].size() == 0 or x[1][0].shape(0) == 0)
  {
    out[cell::type::interval] = xt::xtensor<double, 3>({1, 0, 0});
    return out;
  }

  const xt::xtensor<double, 2> J = {{0., 1.}, {1., 0.}};
  const double detJ = -1.;
  const xt::xtensor<double, 2> K = {{0., 1.}, {1., 0.}};

  xt::xtensor<double, 2> mat = compute_transformation(
      cell::type::triangle, x, M, coeffs, J, detJ, K,
      [](const xt::xtensor<double, 1>& pt) {
        return xt::xtensor<double, 1>({pt[1], pt[0]});
      },
      degree, 1, vs, map_type);

  xt::xtensor<double, 3> transform({1, mat.shape(0), mat.shape(1)});
  xt::view(transform, 0, xt::all(), xt::all()) = mat;

  out[cell::type::interval] = transform;

  return out;
}
//-----------------------------------------------------------------------------
std::map<cell::type, xt::xtensor<double, 3>>
compute_entity_transformations_quadrilateral(
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const int degree, const int vs,
    maps::type map_type)
{
  std::map<cell::type, xt::xtensor<double, 3>> out;
  if (x[1].size() == 0 or x[1][0].shape(0) == 0)
  {
    out[cell::type::interval] = xt::xtensor<double, 3>({1, 0, 0});
    return out;
  }

  const xt::xtensor<double, 2>& pts = x[1][0];
  const xt::xtensor<double, 3>& imat = M[1][0];
  const std::size_t ndofs = imat.shape(0);
  const std::size_t npts = pts.shape(0);
  const int psize = polyset::dim(cell::type::quadrilateral, degree);

  std::size_t dofstart = 0;
  for (std::size_t i = 0; i < M[0].size(); ++i)
    dofstart += M[0][i].shape(0);

  std::size_t total_ndofs = 0;
  for (int d = 0; d <= 3; ++d)
    for (std::size_t i = 0; i < M[d].size(); ++i)
      total_ndofs += M[d][i].shape(0);

  // Map the points to reverse the edge, then tabulate at those points
  xt::xtensor<double, 2> mapped_pts(pts.shape());
  xt::col(mapped_pts, 0) = 1. - xt::col(pts, 0);
  xt::col(mapped_pts, 1) = xt::col(pts, 1);

  xt::xtensor<double, 2> polyset_vals = xt::view(
      polyset::tabulate(cell::type::quadrilateral, degree, 0, mapped_pts), 0,
      xt::all(), xt::all());
  xt::xtensor<double, 3> tabulated_data(
      {npts, total_ndofs, static_cast<std::size_t>(vs)});

  for (int j = 0; j < vs; ++j)
  {
    auto data_view = xt::view(tabulated_data, xt::all(), xt::all(), j);
    xt::xtensor<double, 2> C
        = xt::view(coeffs, xt::all(), xt::range(psize * j, psize * j + psize));
    auto result = math::dot(polyset_vals, xt::transpose(C));
    data_view.assign(result);
  }

  // Pull back
  const xt::xtensor<double, 2> J = {{-1., 0.}, {0., 1.}};
  const double detJ = -1.;
  const xt::xtensor<double, 2> K = {{-1., 0.}, {0., 1.}};
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
  std::array<std::size_t, 3> sh = {1, ndofs, ndofs};
  xt::xtensor<double, 3> transform = xt::zeros<double>(sh);
  for (int i = 0; i < vs; ++i)
  {
    xt::xtensor<double, 2> mat = xt::view(imat, xt::all(), i, xt::all());
    xt::xtensor<double, 2> values = xt::view(dof_data, xt::all(), xt::all(), i);

    xt::view(transform, 1, xt::all(), xt::all()) += math::dot(mat, values);
  }

  out[cell::type::interval] = transform;

  return out;
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
  switch (cell_type)
  {
  case cell::type::interval:
    return std::map<cell::type, xt::xtensor<double, 3>>();
  case cell::type::triangle:
    return compute_entity_transformations_triangle(x, M, coeffs, degree, vs,
                                                   map_type);
  case cell::type::quadrilateral:
    return compute_entity_transformations_quadrilateral(x, M, coeffs, degree,
                                                        vs, map_type);
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
std::vector<int> doftransforms::interval_reflection(int degree)
{
  std::vector<int> perm(std::max(0, degree));
  for (int i = 0; i < degree; ++i)
    perm[i] = degree - 1 - i;
  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> doftransforms::triangle_reflection(int degree)
{
  const int n = degree * (degree + 1) / 2;
  std::vector<int> perm(n);
  int p = 0;
  for (int st = 0; st < degree; ++st)
  {
    int dof = st;
    for (int add = degree; add > st; --add)
    {
      perm[p++] = dof;
      dof += add;
    }
  }

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> doftransforms::triangle_rotation(int degree)
{
  const int n = degree * (degree + 1) / 2;
  std::vector<int> perm(n);
  int p = 0;
  int st = n - 1;
  for (int i = 1; i <= degree; ++i)
  {
    int dof = st;
    for (int sub = i; sub <= degree; ++sub)
    {
      perm[p++] = dof;
      dof -= sub + 1;
    }
    st -= i;
  }

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> doftransforms::quadrilateral_reflection(int degree)
{
  const int n = degree * degree;
  std::vector<int> perm(n);
  int p = 0;
  for (int st = 0; st < degree; ++st)
    for (int i = 0; i < degree; ++i)
      perm[p++] = st + i * degree;

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> doftransforms::quadrilateral_rotation(int degree)
{
  const int n = degree * degree;
  std::vector<int> perm(n);
  int p = 0;
  for (int st = degree - 1; st >= 0; --st)
    for (int i = 0; i < degree; ++i)
      perm[st + degree * i] = p++;

  return perm;
}
//-----------------------------------------------------------------------------
