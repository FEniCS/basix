// Copyright (c) 2021 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "serendipity.h"
#include "element-families.h"
#include "lagrange.h"
#include "lattice.h"
#include "log.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include <numeric>
#include <xtensor/xadapt.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

namespace
{
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_space_2d(int degree)
{
  const std::size_t ndofs = degree == 1 ? 4 : degree * (degree + 3) / 2 + 3;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::quadrilateral, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::quadrilateral, degree, 0, Qpts),
                 0, xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::triangle, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree) polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < nv; ++i)
  {
    auto p_i = xt::col(smaller_polyset_at_Qpts, i);
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
  }

  auto q0 = xt::col(Qpts, 0);
  auto q1 = xt::col(Qpts, 1);
  if (degree == 1)
  {
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(nv, k) = xt::sum(Qwts * q0 * q1 * xt::col(polyset_at_Qpts, k))();
    return wcoeffs;
  }

  xt::xtensor<double, 1> integrand;
  for (std::size_t k = 0; k < psize; ++k)
  {
    auto pk = xt::col(polyset_at_Qpts, k);
    for (std::size_t a = 0; a < 2; ++a)
    {
      auto q_a = xt::col(Qpts, a);
      integrand = Qwts * q0 * q1 * pk;
      for (int i = 1; i < degree; ++i)
        integrand *= q_a;
      wcoeffs(nv + a, k) = xt::sum(integrand)();
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
std::vector<std::array<int, 3>>
serendipity_3d_indices(int total, int linear, std::vector<int> done = {})
{
  if (done.size() == 3)
  {
    int count = 0;
    for (int i = 0; i < 3; ++i)
      if (done[i] == 1)
        ++count;

    if (count >= linear)
      return {{done[0], done[1], done[2]}};
    return {};
  }
  if (done.size() == 2)
  {
    return serendipity_3d_indices(
        total, linear, {done[0], done[1], total - done[0] - done[1]});
  }

  std::vector<int> new_done(done.size() + 1);
  int sum_done = 0;
  for (std::size_t i = 0; i < done.size(); ++i)
  {
    new_done[i] = done[i];
    sum_done += done[i];
  }

  std::vector<std::array<int, 3>> out;
  for (int i = 0; i <= total - sum_done; ++i)
  {
    new_done[done.size()] = i;
    for (std::array<int, 3> j : serendipity_3d_indices(total, linear, new_done))
      out.push_back(j);
  }
  return out;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_space_3d(int degree)
{
  const std::size_t ndofs
      = degree < 4 ? 12 * degree - 4
                   : (degree < 6 ? 3 * degree * degree - 3 * degree + 14
                                 : degree * (degree - 1) * (degree + 1) / 6
                                       + degree * degree + 5 * degree + 4);
  // Number of order (degree) polynomials

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::hexahedron, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::hexahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree) polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize});
  for (std::size_t i = 0; i < nv; ++i)
  {
    auto p_i = xt::col(smaller_polyset_at_Qpts, i);
    for (std::size_t k = 0; k < psize; ++k)
      wcoeffs(i, k) = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
  }

  std::size_t c = nv;
  xt::xtensor<double, 1> integrand;
  std::vector<std::array<int, 3>> indices;
  for (std::size_t s = 1; s <= 3; ++s)
  {
    indices = serendipity_3d_indices(s + degree, s);
    for (std::array<int, 3> i : indices)
    {
      for (std::size_t k = 0; k < psize; ++k)
      {
        integrand = Qwts * xt::col(polyset_at_Qpts, k);
        for (int d = 0; d < 3; ++d)
        {
          auto q_d = xt::col(Qpts, d);
          for (int j = 0; j < i[d]; ++j)
            integrand *= q_d;
        }

        wcoeffs(c, k) = xt::sum(integrand)();
      }
      ++c;
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_div_space_2d(int degree)
{
  const std::size_t ndofs = degree * (degree + 3) + 4;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::quadrilateral, 2 * degree + 2);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts = xt::view(
      polyset::tabulate(cell::type::quadrilateral, degree + 1, 0, Qpts), 0,
      xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::triangle, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * 2});
  for (std::size_t i = 0; i < nv; ++i)
  {
    for (int d = 0; d < 2; ++d)
    {
      auto p_i = xt::col(smaller_polyset_at_Qpts, i);
      for (std::size_t k = 0; k < psize; ++k)
        wcoeffs(d * nv + i, d * psize + k)
            = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
    }
  }

  auto q0 = xt::col(Qpts, 0);
  auto q1 = xt::col(Qpts, 1);

  xt::xtensor<double, 1> integrand;
  for (std::size_t k = 0; k < psize; ++k)
  {
    auto pk = xt::col(polyset_at_Qpts, k);
    for (std::size_t d = 0; d < 2; ++d)
    {
      for (std::size_t a = 0; a < 2; ++a)
      {
        auto q_a = xt::col(Qpts, a);
        integrand = Qwts * pk;
        if (a == 0 and d == 0)
          integrand *= q0;
        else if (a == 0 and d == 1)
          integrand *= (degree + 1) * q1;
        else if (a == 1 and d == 0)
          integrand *= (degree + 1) * q0;
        else if (a == 1 and d == 1)
          integrand *= q1;
        for (int i = 0; i < degree; ++i)
          integrand *= q_a;
        wcoeffs(2 * nv + a, psize * d + k) = xt::sum(integrand)();
      }
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_div_space_3d(int degree)
{
  const std::size_t ndofs = (degree + 1) * (degree * (degree + 5) + 12) / 2;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::hexahedron, 2 * degree + 2);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::hexahedron, degree + 1, 0, Qpts),
                 0, xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * 3});
  for (std::size_t i = 0; i < nv; ++i)
  {
    for (int d = 0; d < 3; ++d)
    {
      auto p_i = xt::col(smaller_polyset_at_Qpts, i);
      for (std::size_t k = 0; k < psize; ++k)
        wcoeffs(d * nv + i, d * psize + k)
            = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
    }
  }

  auto q0 = xt::col(Qpts, 0);
  auto q1 = xt::col(Qpts, 1);
  auto q2 = xt::col(Qpts, 2);

  xt::xtensor<double, 1> integrand;
  for (std::size_t k = 0; k < psize; ++k)
  {
    auto pk = xt::col(polyset_at_Qpts, k);
    for (std::size_t d = 0; d < 3; ++d)
    {
      for (std::size_t a = 0; a < 3; ++a)
      {
        for (int index = 0; index <= degree; ++index)
        {
          auto q_a = xt::col(Qpts, a);
          integrand = Qwts * pk;
          if (a == 0)
          {
            if (d == 0)
              integrand *= -(degree + 2) * q0;
            else if (d == 1)
              integrand *= q1;
            else if (d == 2)
              integrand *= q2;

            for (int i = 0; i < index; ++i)
              integrand *= q1;
            for (int i = 0; i < degree - index; ++i)
              integrand *= q2;
          }
          else if (a == 1)
          {
            if (d == 0)
              integrand *= -q0;
            else if (d == 1)
              integrand *= (degree + 2) * q1;
            else if (d == 2)
              integrand *= -q2;

            for (int i = 0; i < index; ++i)
              integrand *= q0;
            for (int i = 0; i < degree - index; ++i)
              integrand *= q2;
          }
          else if (a == 2)
          {
            if (d == 0)
              integrand *= q0;
            else if (d == 1)
              integrand *= q1;
            else if (d == 2)
              integrand *= -(degree + 2) * q2;

            for (int i = 0; i < index; ++i)
              integrand *= q0;
            for (int i = 0; i < degree - index; ++i)
              integrand *= q1;
          }

          wcoeffs(3 * nv + 3 * index + a, psize * d + k) = xt::sum(integrand)();
        }
      }
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_curl_space_2d(int degree)
{
  const std::size_t ndofs = degree * (degree + 3) + 4;

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::quadrilateral, 2 * degree + 2);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts = xt::view(
      polyset::tabulate(cell::type::quadrilateral, degree + 1, 0, Qpts), 0,
      xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::triangle, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * 2});
  for (std::size_t i = 0; i < nv; ++i)
  {
    for (int d = 0; d < 2; ++d)
    {
      auto p_i = xt::col(smaller_polyset_at_Qpts, i);
      for (std::size_t k = 0; k < psize; ++k)
        wcoeffs(d * nv + i, d * psize + k)
            = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
    }
  }

  auto q0 = xt::col(Qpts, 0);
  auto q1 = xt::col(Qpts, 1);

  xt::xtensor<double, 1> integrand;
  for (std::size_t k = 0; k < psize; ++k)
  {
    auto pk = xt::col(polyset_at_Qpts, k);
    for (std::size_t d = 0; d < 2; ++d)
    {
      for (std::size_t a = 0; a < 2; ++a)
      {
        auto q_a = xt::col(Qpts, a);
        integrand = Qwts * pk;
        if (a == 0 and d == 0)
          integrand *= (degree + 1) * q1;
        else if (a == 0 and d == 1)
          integrand *= -q0;
        else if (a == 1 and d == 0)
          integrand *= q1;
        else if (a == 1 and d == 1)
          integrand *= -(degree + 1) * q0;
        for (int i = 0; i < degree; ++i)
          integrand *= q_a;
        wcoeffs(2 * nv + a, psize * d + k) = xt::sum(integrand)();
      }
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
xt::xtensor<double, 2> make_serendipity_curl_space_3d(int degree)
{
  const std::size_t ndofs = degree <= 3
                                ? 6 * (degree * (degree + 1) + 2)
                                : degree * (degree + 1) * (degree - 1) / 2
                                      + 3 * (degree * (degree + 4) + 3);

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts] = quadrature::make_quadrature(
      "default", cell::type::hexahedron, 2 * degree + 2);
  auto Qwts = xt::adapt(_Qwts);

  xt::xtensor<double, 2> polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::hexahedron, degree + 1, 0, Qpts),
                 0, xt::all(), xt::all());
  xt::xtensor<double, 2> smaller_polyset_at_Qpts
      = xt::view(polyset::tabulate(cell::type::tetrahedron, degree, 0, Qpts), 0,
                 xt::all(), xt::all());

  const std::size_t psize = polyset_at_Qpts.shape(1);
  const std::size_t nv = smaller_polyset_at_Qpts.shape(1);

  // Create coefficients for order (degree) vector polynomials
  xt::xtensor<double, 2> wcoeffs = xt::zeros<double>({ndofs, psize * 3});
  for (std::size_t i = 0; i < nv; ++i)
  {
    for (int d = 0; d < 3; ++d)
    {
      auto p_i = xt::col(smaller_polyset_at_Qpts, i);
      for (std::size_t k = 0; k < psize; ++k)
        wcoeffs(d * nv + i, d * psize + k)
            = xt::sum(Qwts * p_i * xt::col(polyset_at_Qpts, k))();
    }
  }

  auto q0 = xt::col(Qpts, 0);
  auto q1 = xt::col(Qpts, 1);
  auto q2 = xt::col(Qpts, 2);

  xt::xtensor<double, 1> integrand;
  for (std::size_t k = 0; k < psize; ++k)
  {
    auto pk = xt::col(polyset_at_Qpts, k);
    for (std::size_t d = 0; d < 3; ++d)
    {
      for (std::size_t a = 0; a < (degree > 1 ? 3 : 2); ++a)
      {
        for (int index = 0; index <= degree; ++index)
        {
          auto q_a = xt::col(Qpts, a);
          integrand = Qwts * pk;
          if (a == 0)
          {
            if (d == 0)
              integrand *= q1 * q2;
            else if (d == 1)
              integrand *= 0;
            else if (d == 2)
              integrand *= -q0 * q1;

            for (int i = 0; i < index; ++i)
              integrand *= q0;
            for (int i = 0; i < degree - 1 - index; ++i)
              integrand *= q2;
          }
          else if (a == 1)
          {
            if (d == 0)
              integrand *= 0;
            else if (d == 1)
              integrand *= q0 * q2;
            else if (d == 2)
              integrand *= -q0 * q1;

            for (int i = 0; i < index; ++i)
              integrand *= q1;
            for (int i = 0; i < degree - 1 - index; ++i)
              integrand *= q2;
          }
          else if (a == 2)
          {
            if (d == 0)
              integrand *= q1 * q2;
            else if (d == 1)
              integrand *= -q0 * q2;
            else if (d == 2)
              integrand *= 0;

            for (int i = 0; i < index; ++i)
              integrand *= q0;
            for (int i = 0; i < degree - 1 - index; ++i)
              integrand *= q1;
          }

          wcoeffs(3 * nv + 3 * index + a, psize * d + k) = xt::sum(integrand)();
        }
      }
    }
  }
  int c = 3 * nv + (degree > 1 ? 3 : 2) * degree;

  std::vector<std::array<int, 3>> indices;
  for (std::size_t s = 1; s <= 3; ++s)
  {
    indices = serendipity_3d_indices(s + degree + 1, s);
    for (std::array<int, 3> i : indices)
    {
      for (std::size_t k = 0; k < psize; ++k)
      {
        for (int d = 0; d < 3; ++d)
        {
          integrand = Qwts * xt::col(polyset_at_Qpts, k);
          for (int d2 = 0; d2 < 3; ++d2)
          {
            auto q_d2 = xt::col(Qpts, d2);
            if (d == d2)
            {
              integrand *= i[d2];
              for (int j = 0; j < i[d2] - 1; ++j)
                integrand *= q_d2;
            }
            else
            {
              for (int j = 0; j < i[d2]; ++j)
                integrand *= q_d2;
            }
          }

          wcoeffs(c, psize * d + k) = xt::sum(integrand)();
        }
      }
      ++c;
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
FiniteElement basix::create_serendipity(cell::type celltype, int degree)
{
  if (celltype != cell::type::interval and celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::size_t tdim = cell::topological_dimension(celltype);

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  std::vector<xt::xtensor<double, 3>> x(4);
  std::vector<xt::xtensor<double, 4>> M(4);

  const std::size_t vertex_count = cell::sub_entity_count(celltype, 0);
  M[0].resize({vertex_count, 1, vertex_count, 1});
  xt::view(M[0], xt::all(), 0, xt::all(), 0) = xt::eye<double>(vertex_count);

  xt::xtensor<double, 2> points_1d, matrix_1d;
  xt::xtensor<double, 3> edge_transforms, face_transforms;
  if (degree >= 2)
  {
    FiniteElement moment_space = create_dpc(cell::type::interval, degree - 2);
    std::tie(points_1d, matrix_1d)
        = moments::make_integral_moments(moment_space, celltype, 1, quad_deg);
    std::tie(x[1], M[1]) = moments::make_integral_moments_new(
        moment_space, celltype, 1, quad_deg);
    if (tdim > 1)
    {
      edge_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
    }
  }

  xt::xtensor<double, 2> points_2d, matrix_2d;
  if (tdim >= 2 and degree >= 4)
  {
    FiniteElement moment_space
        = create_dpc(cell::type::quadrilateral, degree - 4);
    std::tie(points_2d, matrix_2d)
        = moments::make_integral_moments(moment_space, celltype, 1, quad_deg);
    std::tie(x[2], M[2]) = moments::make_integral_moments_new(
        moment_space, celltype, 1, quad_deg);
    if (tdim > 2)
    {
      face_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
    }
  }

  xt::xtensor<double, 2> points_3d, matrix_3d;
  if (tdim == 3 and degree >= 6)
  {
    std::tie(points_3d, matrix_3d) = moments::make_integral_moments(
        create_dpc(cell::type::hexahedron, degree - 6), celltype, 1, quad_deg);
    std::tie(x[3], M[3]) = moments::make_integral_moments_new(
        create_dpc(cell::type::hexahedron, degree - 6), celltype, 1, quad_deg);
  }

  const std::array<std::size_t, 3> num_pts_dim
      = {points_1d.shape(0), points_2d.shape(0), points_3d.shape(0)};
  std::size_t num_pts
      = std::accumulate(num_pts_dim.begin(), num_pts_dim.end(), 0);

  const std::array<std::size_t, 3> num_mat_dim0
      = {matrix_1d.shape(0), matrix_2d.shape(0), matrix_3d.shape(0)};
  const std::array<std::size_t, 3> num_mat_dim1
      = {matrix_1d.shape(1), matrix_2d.shape(1), matrix_3d.shape(1)};
  std::size_t num_mat0
      = std::accumulate(num_mat_dim0.begin(), num_mat_dim0.end(), 0);
  std::size_t num_mat1
      = std::accumulate(num_mat_dim1.begin(), num_mat_dim1.end(), 0);

  xt::xtensor<double, 2> interpolation_points({vertex_count + num_pts, tdim});
  xt::xtensor<double, 2> interpolation_matrix
      = xt::zeros<double>({vertex_count + num_mat0, vertex_count + num_mat1});

  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);
  xt::view(interpolation_points, xt::range(0, vertex_count), xt::all())
      = geometry;

  x[0].resize({vertex_count, 1, tdim});
  xt::view(x[0], xt::all(), 0, xt::all()) = geometry;

  if (points_1d.size() > 0)
  {
    xt::view(interpolation_points,
             xt::range(vertex_count, vertex_count + num_pts_dim[0]), xt::all())
        = points_1d;
  }

  if (points_2d.size() > 0)
  {
    xt::view(interpolation_points,
             xt::range(vertex_count + num_pts_dim[0],
                       vertex_count + num_pts_dim[0] + num_pts_dim[1]),
             xt::all())
        = points_2d;
  }

  if (points_3d.size() > 0)
  {
    xt::view(interpolation_points,
             xt::range(vertex_count + num_pts_dim[0] + num_pts_dim[1],
                       vertex_count + num_pts_dim[0] + num_pts_dim[1]
                           + num_pts_dim[2]),
             xt::all())
        = points_3d;
  }

  auto r0 = xt::range(0, vertex_count);
  xt::view(interpolation_matrix, r0, r0) = xt::eye<double>(vertex_count);

  xt::view(interpolation_matrix,
           xt::range(vertex_count, vertex_count + num_mat_dim0[0]),
           xt::range(vertex_count, vertex_count + num_mat_dim1[0]))
      = matrix_1d;
  xt::view(interpolation_matrix,
           xt::range(vertex_count + num_mat_dim0[0],
                     vertex_count + num_mat_dim0[0] + num_mat_dim0[1]),
           xt::range(vertex_count + num_mat_dim1[0],
                     vertex_count + num_mat_dim1[0] + +num_mat_dim1[1]))
      = matrix_2d;
  xt::view(interpolation_matrix,
           xt::range(vertex_count + num_mat_dim0[0] + num_mat_dim0[1],
                     vertex_count + num_mat_dim0[0] + num_mat_dim0[1]
                         + num_mat_dim0[2]),
           xt::range(vertex_count + num_mat_dim1[0] + num_mat_dim1[1],
                     vertex_count + num_mat_dim1[0] + +num_mat_dim1[1]
                         + num_mat_dim1[2]))
      = matrix_3d;

  xt::xtensor<double, 2> wcoeffs;
  if (tdim == 1)
    wcoeffs = xt::eye<double>(degree + 1);
  else if (tdim == 2)
    wcoeffs = make_serendipity_space_2d(degree);
  else if (tdim == 3)
    wcoeffs = make_serendipity_space_3d(degree);

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t j = 0; j < topology[0].size(); ++j)
    entity_dofs[0].push_back(1);
  for (std::size_t j = 0; j < topology[1].size(); ++j)
    entity_dofs[1].push_back(num_mat_dim0[0] / topology[1].size());
  if (tdim >= 2)
    for (std::size_t j = 0; j < topology[2].size(); ++j)
      entity_dofs[2].push_back(num_mat_dim0[1] / topology[2].size());
  if (tdim == 3)
    for (std::size_t j = 0; j < topology[3].size(); ++j)
      entity_dofs[3].push_back(num_mat_dim0[2] / topology[3].size());

  const std::size_t ndofs = interpolation_matrix.shape(0);
  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < topology.size() - 1; ++i)
    transform_count += topology[i].size() * i;

  xt::xtensor<double, 3> base_transformations
      = xt::zeros<double>({transform_count, ndofs, ndofs});
  for (std::size_t i = 0; i < base_transformations.shape(0); ++i)
  {
    xt::view(base_transformations, i, xt::all(), xt::all())
        = xt::eye<double>(ndofs);
  }

  if (tdim >= 2 and degree >= 2)
  {
    const int edge_dofs = degree - 1;
    const int num_vertices = topology[0].size();
    const std::size_t num_edges = topology[1].size();
    for (std::size_t edge = 0; edge < num_edges; ++edge)
    {
      const std::size_t start = num_vertices + edge_dofs * edge;
      auto range = xt::range(start, start + edge_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(edge_transforms, 0, xt::all(), xt::all());
    }
    if (tdim == 3 and degree >= 4)
    {
      const std::size_t face_dofs = face_transforms.shape(1);
      const std::size_t num_faces = topology[2].size();
      for (std::size_t face = 0; face < num_faces; ++face)
      {
        const std::size_t start
            = num_vertices + num_edges * edge_dofs + face * face_dofs;
        auto range = xt::range(start, start + face_dofs);
        xt::view(base_transformations, num_edges + 2 * face, range, range)
            = xt::view(face_transforms, 0, xt::all(), xt::all());
        xt::view(base_transformations, num_edges + 2 * face + 1, range, range)
            = xt::view(face_transforms, 1, xt::all(), xt::all());
      }
    }
  }

  xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients(celltype, wcoeffs, M, x, degree);
  return FiniteElement(element::family::Serendipity, celltype, degree, {1},
                       coeffs, entity_dofs, base_transformations,
                       interpolation_points, interpolation_matrix,
                       maps::type::identity);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_serendipity_div(cell::type celltype, int degree)
{
  if (celltype != cell::type::interval and celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::size_t tdim = cell::topological_dimension(celltype);

  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::quadrilateral;

  // Number of dofs and interpolation points
  int quad_deg = 5 * degree;

  xt::xtensor<double, 2> points_facet, matrix_facet;
  xt::xtensor<double, 3> facet_transforms;

  FiniteElement facet_moment_space = create_dpc(facettype, degree);
  std::tie(points_facet, matrix_facet) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  auto [x1, M1] = moments::make_normal_integral_moments_new(
      facet_moment_space, celltype, tdim, quad_deg);
  std::vector<xt::xtensor<double, 3>> x = {x1};
  std::vector<xt::xtensor<double, 4>> M = {M1};

  if (tdim > 1)
  {
    facet_transforms
        = moments::create_normal_moment_dof_transformations(facet_moment_space);
  }

  xt::xtensor<double, 2> points_cell = {};
  xt::xtensor<double, 2> matrix_cell = {};
  if (tdim >= 2 and degree >= 2)
  {
    FiniteElement cell_moment_space = create_dpc(celltype, degree - 2);
    std::tie(points_cell, matrix_cell) = moments::make_integral_moments(
        cell_moment_space, celltype, tdim, quad_deg);
    auto [x2, M2] = moments::make_integral_moments_new(
        cell_moment_space, celltype, tdim, quad_deg);
    x.push_back(x2);
    M.push_back(M2);
  }

  // Interpolation points and matrix
  xt::xtensor<double, 2> interpolation_points, interpolation_matrix;
  std::tie(interpolation_points, interpolation_matrix)
      = combine_interpolation_data(points_facet, points_cell, {}, matrix_facet,
                                   matrix_cell, {}, tdim, tdim);

  const int facet_dofs = facet_moment_space.dim();
  const int cell_dofs = matrix_cell.shape(0);
  const int vertex_dofs = tdim == 1 ? facet_dofs : 0;
  const int edge_dofs = tdim == 1 ? cell_dofs : (tdim == 2 ? facet_dofs : 0);
  const int face_dofs = tdim == 2 ? cell_dofs : (tdim == 3 ? facet_dofs : 0);
  const int volume_dofs = tdim == 3 ? cell_dofs : 0;

  std::vector<std::vector<int>> entity_dofs(topology.size());
  for (std::size_t j = 0; j < topology[0].size(); ++j)
    entity_dofs[0].push_back(vertex_dofs);
  for (std::size_t j = 0; j < topology[1].size(); ++j)
    entity_dofs[1].push_back(edge_dofs);
  if (tdim >= 2)
    for (std::size_t j = 0; j < topology[2].size(); ++j)
      entity_dofs[2].push_back(face_dofs);
  if (tdim == 3)
    for (std::size_t j = 0; j < topology[3].size(); ++j)
      entity_dofs[3].push_back(volume_dofs);

  const std::size_t ndofs = interpolation_matrix.shape(0);

  xt::xtensor<double, 2> wcoeffs;
  if (tdim == 1)
    wcoeffs = xt::eye<double>(degree + 1);
  else if (tdim == 2)
    wcoeffs = make_serendipity_div_space_2d(degree);
  else if (tdim == 3)
    wcoeffs = make_serendipity_div_space_3d(degree);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;

  const int facet_count = topology[tdim - 1].size();

  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  if (tdim == 2)
  {
    for (int edge = 0; edge < facet_count; ++edge)
    {
      const std::size_t start = facet_dofs * edge;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, edge, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
    }
  }
  else if (tdim == 3)
  {
    const int edge_count = 12;
    for (int face = 0; face < facet_count; ++face)
    {
      const std::size_t start = facet_dofs * face;
      const std::size_t p = edge_count + 2 * face;
      auto range = xt::range(start, start + facet_dofs);
      xt::view(base_transformations, p, range, range)
          = xt::view(facet_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, p + 1, range, range)
          = xt::view(facet_transforms, 1, xt::all(), xt::all());
    }
  }

  xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients(celltype, wcoeffs, M, x, degree + 1);

  return FiniteElement(element::family::BDM, celltype, degree + 1, {tdim},
                       coeffs, entity_dofs, base_transformations,
                       interpolation_points, interpolation_matrix,
                       maps::type::contravariantPiola);
}
//-----------------------------------------------------------------------------
FiniteElement basix::create_serendipity_curl(cell::type celltype, int degree)
{
  if (celltype != cell::type::interval and celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  auto [Qpts, _Qwts]
      = quadrature::make_quadrature("default", celltype, 2 * degree);
  auto Qwts = xt::adapt(_Qwts);
  xt::xtensor<double, 2> polyset_at_Qpts = xt::view(
      polyset::tabulate(celltype, degree, 0, Qpts), 0, xt::all(), xt::all());

  const int edge_count = tdim == 2 ? 4 : 12;
  const int edge_dofs = polyset::dim(cell::type::interval, degree);
  const int face_count = tdim == 2 ? 1 : 6;
  const int face_dofs
      = (tdim < 2 or degree < 2)
            ? 0
            : 2 * polyset::dim(cell::type::triangle, degree - 2);
  const int volume_count = tdim == 2 ? 0 : 1;
  const int volume_dofs
      = (tdim < 3 or degree < 4)
            ? 0
            : 3 * polyset::dim(cell::type::tetrahedron, degree - 4);

  const std::size_t ndofs = edge_count * edge_dofs + face_count * face_dofs
                            + volume_count * volume_dofs;

  xt::xtensor<double, 2> wcoeffs;
  if (tdim == 1)
    wcoeffs = xt::eye<double>(degree + 1);
  else if (tdim == 2)
    wcoeffs = make_serendipity_curl_space_2d(degree);
  else if (tdim == 3)
    wcoeffs = make_serendipity_curl_space_3d(degree);

  FiniteElement edge_moment_space = create_dpc(cell::type::interval, degree);

  xt::xtensor<double, 2> points_1d, matrix_1d;
  std::tie(points_1d, matrix_1d) = moments::make_tangent_integral_moments(
      edge_moment_space, celltype, tdim, 2 * degree + 2);

  auto [x1, M1] = moments::make_tangent_integral_moments_new(
      edge_moment_space, celltype, tdim, 2 * degree + 2);
  std::vector<xt::xtensor<double, 3>> x = {x1};
  std::vector<xt::xtensor<double, 4>> M = {M1};
  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(edge_moment_space);

  // Add integral moments on interior
  xt::xtensor<double, 2> points_2d, matrix_2d, points_3d, matrix_3d;
  xt::xtensor<double, 3> face_transforms;
  if (degree >= 2)
  {
    // Face integral moment
    FiniteElement moment_space
        = create_dpc(cell::type::quadrilateral, degree - 2);
    std::tie(points_2d, matrix_2d) = moments::make_integral_moments(
        moment_space, celltype, tdim, 2 * degree);
    auto [x2, M2] = moments::make_integral_moments_new(moment_space, celltype,
                                                       tdim, 2 * degree);
    x.push_back(x2);
    M.push_back(M2);

    if (tdim == 3)
    {
      face_transforms
          = moments::create_moment_dof_transformations(moment_space);

      if (degree >= 4)
      {
        // Interior integral moment
        std::tie(points_3d, matrix_3d) = moments::make_integral_moments(
            create_dpc(cell::type::hexahedron, degree - 4), celltype, tdim,
            2 * degree - 3);
        auto [x3, M3] = moments::make_integral_moments_new(
            create_dpc(cell::type::hexahedron, degree - 4), celltype, tdim,
            2 * degree - 3);
        x.push_back(x3);
        M.push_back(M3);
      }
    }
  }

  // Interpolation points and matrix
  xt::xtensor<double, 2> points, matrix;
  std::tie(points, matrix)
      = combine_interpolation_data(points_1d, points_2d, points_3d, matrix_1d,
                                   matrix_2d, matrix_3d, tdim, tdim);

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::size_t transform_count = 0;
  for (std::size_t i = 1; i < tdim; ++i)
    transform_count += topology[i].size() * i;
  auto base_transformations
      = xt::tile(xt::expand_dims(xt::eye<double>(ndofs), 0), transform_count);
  for (int edge = 0; edge < edge_count; ++edge)
  {
    const std::size_t start = edge_dofs * edge;
    auto range = xt::range(start, start + edge_dofs);
    xt::view(base_transformations, edge, range, range)
        = xt::view(edge_transforms, 0, xt::all(), xt::all());
  }


  if (tdim == 3 and degree > 1)
  {
    for (int face = 0; face < face_count; ++face)
    {
      const std::size_t start = edge_dofs * edge_count + face_dofs * face;
      const std::size_t p = edge_count + 2 * face;
      auto range = xt::range(start, start + face_dofs);
      xt::view(base_transformations, p, range, range)
          = xt::view(face_transforms, 0, xt::all(), xt::all());
      xt::view(base_transformations, p + 1, range, range)
          = xt::view(face_transforms, 1, xt::all(), xt::all());
    }
  }

  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  entity_dofs[1].resize(topology[1].size(), edge_dofs);
  entity_dofs[2].resize(topology[2].size(), face_dofs);
  if (tdim == 3)
    entity_dofs[3].resize(topology[3].size(), volume_dofs);

  xt::xtensor<double, 3> coeffs
      = compute_expansion_coefficients(celltype, wcoeffs, M, x, degree + 1);

  return FiniteElement(element::family::N2E, celltype, degree + 1, {tdim},
                       coeffs, entity_dofs, base_transformations, points,
                       matrix, maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
