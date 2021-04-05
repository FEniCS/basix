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
#include <xtensor/xbuilder.hpp>
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

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  // dim 0 (vertices)
  const xt::xtensor<double, 2> geometry = cell::geometry(celltype);
  const std::size_t num_vertices = geometry.shape(0);
  M[0] = std::vector<xt::xtensor<double, 3>>(num_vertices,
                                             xt::ones<double>({1, 1, 1}));
  x[0].resize(geometry.shape(0));
  for (std::size_t i = 0; i < x[0].size(); ++i)
  {
    x[0][i] = xt::reshape_view(
        xt::row(geometry, i), {static_cast<std::size_t>(1), geometry.shape(1)});
  }

  xt::xtensor<double, 3> edge_transforms, face_transforms;
  if (degree >= 2)
  {
    FiniteElement moment_space = create_dpc(cell::type::interval, degree - 2);
    std::tie(x[1], M[1])
        = moments::make_integral_moments(moment_space, celltype, 1, quad_deg);

    if (tdim > 1)
    {
      edge_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
    }
  }

  if (tdim >= 2 and degree >= 4)
  {
    FiniteElement moment_space
        = create_dpc(cell::type::quadrilateral, degree - 4);
    std::tie(x[2], M[2])
        = moments::make_integral_moments(moment_space, celltype, 1, quad_deg);
    if (tdim > 2)
    {
      face_transforms
          = moments::create_dot_moment_dof_transformations(moment_space);
    }
  }

  if (tdim == 3 and degree >= 6)
  {
    std::tie(x[3], M[3]) = moments::make_integral_moments(
        create_dpc(cell::type::hexahedron, degree - 6), celltype, 1, quad_deg);
  }

  xt::xtensor<double, 2> wcoeffs;
  if (tdim == 1)
    wcoeffs = xt::eye<double>(degree + 1);
  else if (tdim == 2)
    wcoeffs = make_serendipity_space_2d(degree);
  else if (tdim == 3)
    wcoeffs = make_serendipity_space_3d(degree);

  std::vector<std::vector<int>> entity_dofs(topology.size());
  entity_dofs[0].resize(topology[0].size(), 0);
  for (std::size_t v = 0; v < M[0].size(); ++v)
    entity_dofs[0][v] = 1;

  entity_dofs[1].resize(topology[1].size(), 0);
  for (std::size_t e = 0; e < M[1].size(); ++e)
    entity_dofs[1][e] = M[1][e].shape(0);

  if (tdim >= 2)
  {
    entity_dofs[2].resize(topology[2].size(), 0);
    for (std::size_t e = 0; e < M[2].size(); ++e)
      entity_dofs[2][e] = M[2][e].shape(0);
  }

  if (tdim == 3)
  {
    entity_dofs[3].resize(topology[3].size(), 0);
    for (std::size_t e = 0; e < M[3].size(); ++e)
      entity_dofs[3][e] = M[3][e].shape(0);
  }

  std::size_t ndofs = 0;
  for (auto& Md : M)
    for (auto& Me : Md)
      ndofs += Me.shape(0);

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

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[0], M[1], M[2], M[3]}, {x[0], x[1], x[2], x[3]},
      degree);

  return FiniteElement(element::family::Serendipity, celltype, degree, {1},
                       coeffs, entity_dofs, base_transformations, x, M,
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

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  xt::xtensor<double, 3> facet_transforms;

  FiniteElement facet_moment_space = create_dpc(facettype, degree);
  std::tie(x[tdim - 1], M[tdim - 1]) = moments::make_normal_integral_moments(
      facet_moment_space, celltype, tdim, quad_deg);

  if (tdim > 1)
  {
    facet_transforms
        = moments::create_normal_moment_dof_transformations(facet_moment_space);
  }

  int cell_dofs = 0;
  if (tdim >= 2 and degree >= 2)
  {
    FiniteElement cell_moment_space = create_dpc(celltype, degree - 2);
    std::tie(x[tdim], M[tdim]) = moments::make_integral_moments(
        cell_moment_space, celltype, tdim, quad_deg);
    cell_dofs = M[tdim][0].shape(0);
  }

  const int facet_dofs = facet_moment_space.dim();
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

  std::size_t ndofs = 0;
  for (auto& Md : M)
    for (auto& Me : Md)
      ndofs += Me.shape(0);

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

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[tdim - 1], M[tdim]}, {x[tdim - 1], x[tdim]},
      degree + 1);

  return FiniteElement(element::family::BDM, celltype, degree + 1, {tdim},
                       coeffs, entity_dofs, base_transformations, x, M,
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

  std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  FiniteElement edge_moment_space = create_dpc(cell::type::interval, degree);

  std::tie(x[1], M[1]) = moments::make_tangent_integral_moments(
      edge_moment_space, celltype, tdim, 2 * degree + 2);
  xt::xtensor<double, 3> edge_transforms
      = moments::create_tangent_moment_dof_transformations(edge_moment_space);

  // Add integral moments on interior
  xt::xtensor<double, 3> face_transforms;
  if (degree >= 2)
  {
    // Face integral moment
    FiniteElement moment_space
        = create_dpc(cell::type::quadrilateral, degree - 2);
    std::tie(x[2], M[2]) = moments::make_integral_moments(
        moment_space, celltype, tdim, 2 * degree);
    if (tdim == 3)
    {
      face_transforms
          = moments::create_moment_dof_transformations(moment_space);
      if (degree >= 4)
      {
        // Interior integral moment
        std::tie(x[3], M[3]) = moments::make_integral_moments(
            create_dpc(cell::type::hexahedron, degree - 4), celltype, tdim,
            2 * degree - 3);
      }
    }
  }

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

  xt::xtensor<double, 3> coeffs = compute_expansion_coefficients(
      celltype, wcoeffs, {M[1], M[2], M[3]}, {x[1], x[2], x[3]}, degree + 1);

  return FiniteElement(element::family::N2E, celltype, degree + 1, {tdim},
                       coeffs, entity_dofs, base_transformations, x, M,
                       maps::type::covariantPiola);
}
//-----------------------------------------------------------------------------
