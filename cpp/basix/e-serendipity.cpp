// Copyright (c) 2021 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-serendipity.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polynomials.h"
#include "polyset.h"
#include "quadrature.h"

using namespace basix;

namespace
{
//----------------------------------------------------------------------------
impl::mdarray2_t make_serendipity_space_2d(int degree)
{
  const std::size_t ndofs = degree == 1 ? 4 : degree * (degree + 3) / 2 + 3;

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::quadrilateral, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_Pq, shape]
      = polyset::tabulate(cell::type::quadrilateral, degree, 0, pts);
  impl::cmdspan3_t Pq(_Pq.data(), shape);

  const std::size_t psize = Pq.extent(1);

  // Create coefficients for order (degree) polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize);
  int row_n = 0;
  for (int i = 0; i <= degree; ++i)
    for (int j = 0; j <= degree - i; ++j)
      wcoeffs(row_n++, i * (degree + 1) + j) = 1;

  if (degree == 1)
  {
    for (std::size_t i = 0; i < psize; ++i)
    {
      wcoeffs(row_n, i) = 0.0;
      for (std::size_t j = 0; j < wts.size(); ++j)
        wcoeffs(row_n, i) += wts[j] * pts(j, 0) * pts(j, 1) * Pq(0, i, j);
    }
    return wcoeffs;
  }

  std::vector<double> integrand(wts.size());
  for (std::size_t k = 0; k < psize; ++k)
  {
    for (std::size_t a = 0; a < 2; ++a)
    {
      for (std::size_t i = 0; i < integrand.size(); ++i)
        integrand[i] = wts[i] * pts(i, 0) * pts(i, 1) * Pq(0, k, i);

      for (int i = 1; i < degree; ++i)
        for (std::size_t j = 0; j < integrand.size(); ++j)
          integrand[j] *= pts(j, a);

      wcoeffs(row_n + a, k) = 0;
      for (std::size_t i = 0; i < integrand.size(); ++i)
        wcoeffs(row_n + a, k) += integrand[i];
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
    else
      return {};
  }
  else if (done.size() == 2)
  {
    return serendipity_3d_indices(
        total, linear, {done[0], done[1], total - done[0] - done[1]});
  }

  std::vector<int> new_done(done.size() + 1);
  std::copy(done.begin(), done.end(), new_done.begin());
  const int sum_done = std::reduce(done.begin(), done.end());

  std::vector<std::array<int, 3>> out;
  for (int i = 0; i <= total - sum_done; ++i)
  {
    new_done.back() = i;
    for (std::array<int, 3> j : serendipity_3d_indices(total, linear, new_done))
      out.push_back(j);
  }

  return out;
}
//----------------------------------------------------------------------------
impl::mdarray2_t make_serendipity_space_3d(int degree)
{
  const std::size_t ndofs
      = degree == 0
            ? 1
            : (degree < 4
                   ? 12 * degree - 4
                   : (degree < 6 ? 3 * degree * degree - 3 * degree + 14
                                 : degree * (degree - 1) * (degree + 1) / 6
                                       + degree * degree + 5 * degree + 4));
  // Number of order (degree) polynomials

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::hexahedron, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  const auto [_Ph, shape]
      = polyset::tabulate(cell::type::hexahedron, degree, 0, pts);
  impl::cmdspan3_t Ph(_Ph.data(), shape);
  const std::size_t psize = Ph.extent(1);

  // Create coefficients for order (degree) polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize);
  int row_n = 0;
  for (int i = 0; i <= degree; ++i)
    for (int j = 0; j <= degree - i; ++j)
      for (int k = 0; k <= degree - i - j; ++k)
        wcoeffs(row_n++, i * (degree + 1) * (degree + 1) + j * (degree + 1) + k)
            = 1;

  std::vector<double> integrand(wts.size());
  std::vector<std::array<int, 3>> indices;
  for (std::size_t s = 1; s <= 3; ++s)
  {
    indices = serendipity_3d_indices(s + degree, s);
    for (std::array<int, 3> i : indices)
    {
      for (std::size_t k = 0; k < psize; ++k)
      {
        for (std::size_t i = 0; i < integrand.size(); ++i)
          integrand[i] = wts[i] * Ph(0, k, i);

        for (int d = 0; d < 3; ++d)
        {
          for (int j = 0; j < i[d]; ++j)
            for (std::size_t l = 0; l < integrand.size(); ++l)
              integrand[l] *= pts(l, d);
        }

        wcoeffs(row_n, k) = 0;
        for (std::size_t j = 0; j < integrand.size(); ++j)
          wcoeffs(row_n, k) += integrand[j];
      }
      ++row_n;
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
impl::mdarray2_t make_serendipity_div_space_2d(int degree)
{
  const std::size_t ndofs = degree * (degree + 3) + 4;

  // Evaluate the expansion polynomials at the quadrature points
  auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::quadrilateral, 2 * degree + 2);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_Pq, shape]
      = polyset::tabulate(cell::type::quadrilateral, degree + 1, 0, pts);
  impl::cmdspan3_t Pq(_Pq.data(), shape);

  const std::size_t psize = Pq.extent(1);
  const std::size_t nv = polyset::dim(cell::type::triangle, degree);

  // Create coefficients for order (degree) vector polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * 2);
  int row_n = 0;
  for (int i = 0; i <= degree; ++i)
    for (int j = 0; j <= degree - i; ++j)
      for (int d = 0; d < 2; ++d)
        wcoeffs(row_n++, d * psize + i * (degree + 2) + j) = 1;

  std::vector<double> integrand(wts.size());
  for (std::size_t k = 0; k < psize; ++k)
  {
    for (std::size_t d = 0; d < 2; ++d)
    {
      for (std::size_t a = 0; a < 2; ++a)
      {
        for (std::size_t i = 0; i < integrand.size(); ++i)
          integrand[i] = wts[i] * Pq(0, k, i);

        if (a == 0 and d == 0)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= pts(i, 0);
        }
        else if (a == 0 and d == 1)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= (degree + 1) * pts(i, 1);
        }
        else if (a == 1 and d == 0)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= (degree + 1) * pts(i, 0);
        }
        else if (a == 1 and d == 1)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= pts(i, 1);
        }

        for (int i = 0; i < degree; ++i)
          for (std::size_t j = 0; j < integrand.size(); ++j)
            integrand[j] *= pts(j, a);

        wcoeffs(2 * nv + a, psize * d + k)
            = std::reduce(integrand.begin(), integrand.end(), 0.0);
      }
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
impl::mdarray2_t make_serendipity_div_space_3d(int degree)
{
  const std::size_t ndofs = (degree + 1) * (degree * (degree + 5) + 12) / 2;

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::hexahedron, 2 * degree + 2);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  const auto [_Pq, shape]
      = polyset::tabulate(cell::type::hexahedron, degree + 1, 0, pts);
  impl::cmdspan3_t Pq(_Pq.data(), shape);

  const std::size_t psize = Pq.extent(1);
  const std::size_t nv = polyset::dim(cell::type::tetrahedron, degree);

  // Create coefficients for order (degree) vector polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * 3);
  int row_n = 0;
  for (int i = 0; i <= degree; ++i)
  {
    for (int j = 0; j <= degree - i; ++j)
    {
      for (int k = 0; k <= degree - i - j; ++k)
      {
        for (int d = 0; d < 3; ++d)
        {
          wcoeffs(row_n++, d * psize + i * (degree + 2) * (degree + 2)
                               + j * (degree + 2) + k)
              = 1;
        }
      }
    }
  }

  std::vector<double> integrand(wts.size());
  for (std::size_t k = 0; k < psize; ++k)
  {
    for (std::size_t d = 0; d < 3; ++d)
    {
      for (std::size_t a = 0; a < 3; ++a)
      {
        for (int index = 0; index <= degree; ++index)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] = wts[i] * Pq(0, k, i);

          if (a == 0)
          {
            if (d == 0)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -(degree + 2) * pts(i, 0);
            }
            else if (d == 1)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1);
            }
            else if (d == 2)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 2);
            }

            for (int i = 0; i < index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1);
            }
            for (int i = 0; i < degree - index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 2);
            }
          }
          else if (a == 1)
          {
            if (d == 0)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -pts(i, 0);
            }
            else if (d == 1)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= (degree + 2) * pts(i, 1);
            }
            else if (d == 2)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -pts(i, 2);
            }

            for (int i = 0; i < index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 0);
            }
            for (int i = 0; i < degree - index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 2);
            }
          }
          else if (a == 2)
          {
            if (d == 0)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 0);
            }
            else if (d == 1)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1);
            }
            else if (d == 2)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -(degree + 2) * pts(i, 2);
            }

            for (int i = 0; i < index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 0);
            }
            for (int i = 0; i < degree - index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1);
            }
          }

          wcoeffs(3 * nv + 3 * index + a, psize * d + k)
              = std::reduce(integrand.begin(), integrand.end(), 0.0);
        }
      }
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
impl::mdarray2_t make_serendipity_curl_space_2d(int degree)
{
  const std::size_t ndofs = degree * (degree + 3) + 4;

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::quadrilateral, 2 * degree + 2);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_Pq, shape]
      = polyset::tabulate(cell::type::quadrilateral, degree + 1, 0, pts);
  impl::cmdspan3_t Pq(_Pq.data(), shape);

  const std::size_t psize = Pq.extent(1);
  const std::size_t nv = polyset::dim(cell::type::triangle, degree);

  // Create coefficients for order (degree) vector polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * 2);
  int row_n = 0;
  for (int i = 0; i <= degree; ++i)
    for (int j = 0; j <= degree - i; ++j)
      for (int d = 0; d < 2; ++d)
        wcoeffs(row_n++, d * psize + i * (degree + 2) + j) = 1;

  std::vector<double> integrand(wts.size());
  for (std::size_t k = 0; k < psize; ++k)
  {
    for (std::size_t d = 0; d < 2; ++d)
    {
      for (std::size_t a = 0; a < 2; ++a)
      {
        for (std::size_t i = 0; i < integrand.size(); ++i)
          integrand[i] = wts[i] * Pq(0, k, i);

        if (a == 0 and d == 0)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= (degree + 1) * pts(i, 1);
        }
        else if (a == 0 and d == 1)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= -pts(i, 0);
        }
        else if (a == 1 and d == 0)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= pts(i, 1);
        }
        else if (a == 1 and d == 1)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= -(degree + 1) * pts(i, 0);
        }

        for (int i = 0; i < degree; ++i)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] *= pts(i, a);
        }

        wcoeffs(2 * nv + a, psize * d + k)
            = std::reduce(integrand.begin(), integrand.end(), 0.0);
      }
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
impl::mdarray2_t make_serendipity_curl_space_3d(int degree)
{
  const std::size_t ndofs = degree <= 3
                                ? 6 * (degree * (degree + 1) + 2)
                                : degree * (degree + 1) * (degree - 1) / 2
                                      + 3 * (degree * (degree + 4) + 3);

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, cell::type::hexahedron, 2 * degree + 2);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_Pq, shape]
      = polyset::tabulate(cell::type::hexahedron, degree + 1, 0, pts);
  impl::cmdspan3_t Pq(_Pq.data(), shape);

  const std::size_t psize = Pq.extent(1);
  const std::size_t nv = polyset::dim(cell::type::tetrahedron, degree);

  // Create coefficients for order (degree) vector polynomials
  impl::mdarray2_t wcoeffs(ndofs, psize * 3);
  int row_n = 0;
  for (int i = 0; i <= degree; ++i)
  {
    for (int j = 0; j <= degree - i; ++j)
    {
      for (int k = 0; k <= degree - i - j; ++k)
      {
        for (int d = 0; d < 3; ++d)
        {
          wcoeffs(row_n++, d * psize + i * (degree + 2) * (degree + 2)
                               + j * (degree + 2) + k)
              = 1;
        }
      }
    }
  }

  std::vector<double> integrand(wts.size());
  for (std::size_t k = 0; k < psize; ++k)
  {
    for (std::size_t d = 0; d < 3; ++d)
    {
      for (std::size_t a = 0; a < (degree > 1 ? 3 : 2); ++a)
      {
        for (int index = 0; index <= degree; ++index)
        {
          for (std::size_t i = 0; i < integrand.size(); ++i)
            integrand[i] = wts[i] * Pq(0, k, i);

          if (a == 0)
          {
            if (d == 0)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1) * pts(i, 2);
            }
            else if (d == 1)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= 0;
            }
            else if (d == 2)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -pts(i, 0) * pts(i, 1);
            }
            for (int i = 0; i < index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 0);
            }
            for (int i = 0; i < degree - 1 - index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 2);
            }
          }
          else if (a == 1)
          {
            if (d == 0)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= 0;
            }
            else if (d == 1)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 0) * pts(i, 2);
            }
            else if (d == 2)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -pts(i, 0) * pts(i, 1);
            }

            for (int i = 0; i < index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1);
            }
            for (int i = 0; i < degree - 1 - index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 2);
            }
          }
          else if (a == 2)
          {
            if (d == 0)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1) * pts(i, 2);
            }
            else if (d == 1)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= -pts(i, 0) * pts(i, 2);
            }
            else if (d == 2)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= 0;
            }
            for (int i = 0; i < index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 0);
            }
            for (int i = 0; i < degree - 1 - index; ++i)
            {
              for (std::size_t i = 0; i < integrand.size(); ++i)
                integrand[i] *= pts(i, 1);
            }
          }

          wcoeffs(3 * nv + 3 * index + a, psize * d + k)
              = std::reduce(integrand.begin(), integrand.end(), 0.0);
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
          for (std::size_t j = 0; j < integrand.size(); ++j)
            integrand[j] = wts[j] * Pq(0, k, j);
          for (int d2 = 0; d2 < 3; ++d2)
          {
            if (d == d2)
            {
              for (std::size_t j = 0; j < integrand.size(); ++j)
                integrand[j] *= i[d2];
              for (int j = 0; j < i[d2] - 1; ++j)
              {
                for (std::size_t j = 0; j < integrand.size(); ++j)
                  integrand[j] *= pts(j, d2);
              }
            }
            else
            {
              for (int j = 0; j < i[d2]; ++j)
              {
                for (std::size_t j = 0; j < integrand.size(); ++j)
                  integrand[j] *= pts(j, d2);
              }
            }
          }

          wcoeffs(c, psize * d + k)
              = std::reduce(integrand.begin(), integrand.end(), 0.0);
        }
      }
      ++c;
    }
  }

  return wcoeffs;
}
//----------------------------------------------------------------------------
FiniteElement create_legendre_dpc(cell::type celltype, int degree,
                                  bool discontinuous)
{
  if (!discontinuous)
    throw std::runtime_error("Legendre variant must be discontinuous");

  cell::type simplex_type;
  switch (celltype)
  {
  case cell::type::quadrilateral:
    simplex_type = cell::type::triangle;
    break;
  case cell::type::hexahedron:
    simplex_type = cell::type::tetrahedron;
    break;
  default:
    throw std::runtime_error("Invalid cell type");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t psize = polyset::dim(celltype, degree);
  const std::size_t ndofs = polyset::dim(simplex_type, degree);

  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, celltype, degree * 2);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());

  // Evaluate moment space at quadrature points
  const auto [_phi, shape] = polynomials::tabulate(polynomials::type::legendre,
                                                   celltype, degree, pts);
  impl::cmdspan2_t phi(_phi.data(), shape);

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  for (std::size_t i = 0; i < tdim; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, 1, 0, 1));
  }

  x[tdim].emplace_back(_pts, pts.extent(0), pts.extent(1));
  auto& _M = M[tdim].emplace_back(ndofs, 1, pts.extent(0), 1);

  impl::mdarray2_t wcoeffs(ndofs, psize);
  if (celltype == cell::type::quadrilateral)
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (std::size_t k = 0; k < wts.size(); ++k)
          _M(row_n, 0, k, 0) = phi(i * (degree + 1) + j, k) * wts[k];
        wcoeffs(row_n, i * (degree + 1) + j) = 1;
        ++row_n;
      }
    }
  }
  else
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (int k = 0; k <= degree - i - j; ++k)
        {
          for (std::size_t l = 0; l < wts.size(); ++l)
          {
            _M(row_n, 0, l, 0)
                = phi(i * (degree + 1) * (degree + 1) + j * (degree + 1) + k, l)
                  * wts[l];
          }
          wcoeffs(row_n, i * (degree + 1) * (degree + 1) + j * (degree + 1) + k)
              = 1;
          ++row_n;
        }
      }
    }
  }

  return FiniteElement(element::family::DPC, celltype, degree, {},
                       impl::mdspan2_t(wcoeffs.data(), wcoeffs.extents()),
                       impl::to_mdspan(x), impl::to_mdspan(M), 0,
                       maps::type::identity, discontinuous, degree, degree,
                       element::dpc_variant::legendre);
}
//-----------------------------------------------------------------------------
impl::mdarray2_t make_dpc_points(cell::type celltype, int degree,
                                 element::dpc_variant variant)
{
  auto to_mdspan
      = [](auto& x, auto shape) { return impl::cmdspan2_t(x.data(), shape); };
  auto to_mdarray = [](auto& x, auto shape)
  { return impl::mdarray2_t(x, shape[0], shape[1]); };

  if (degree == 0)
  {
    const auto [data, shape]
        = lattice::create(celltype, 0, lattice::type::equispaced, true);
    return to_mdarray(data, shape);
  }

  if (variant == element::dpc_variant::simplex_equispaced
      or variant == element::dpc_variant::simplex_gll)
  {
    lattice::type latticetype;
    lattice::simplex_method latticesm = lattice::simplex_method::isaac;
    if (variant == element::dpc_variant::simplex_equispaced)
      latticetype = lattice::type::equispaced;
    else if (variant == element::dpc_variant::simplex_gll)
      latticetype = lattice::type::gll;

    switch (celltype)
    {
    case cell::type::quadrilateral:
    {
      const auto [data, shape] = lattice::create(cell::type::triangle, degree,
                                                 latticetype, true, latticesm);
      return to_mdarray(data, shape);
    }
    case cell::type::hexahedron:
    {
      const auto [data, shape] = lattice::create(
          cell::type::tetrahedron, degree, latticetype, true, latticesm);
      return to_mdarray(data, shape);
    }
    default:
      throw std::runtime_error("Invalid cell type");
    }
  }
  else if (variant == element::dpc_variant::horizontal_equispaced
           or variant == element::dpc_variant::horizontal_gll)
  {
    lattice::type latticetype;
    if (variant == element::dpc_variant::horizontal_equispaced)
      latticetype = lattice::type::equispaced;
    else if (variant == element::dpc_variant::horizontal_gll)
      latticetype = lattice::type::gll;

    switch (celltype)
    {
    case cell::type::quadrilateral:
    {
      impl::mdarray2_t pts((degree + 2) * (degree + 1) / 2, 2);
      std::size_t n = 0;
      for (int j = 0; j <= degree; ++j)
      {
        const auto [data, shape] = lattice::create(
            cell::type::interval, degree - j, latticetype, true);
        auto interval_pts = to_mdspan(data, shape);
        for (int i = 0; i <= degree - j; ++i)
        {
          pts(n, 0) = interval_pts(i, 0);
          pts(n, 1) = j % 2 == 0
                          ? static_cast<double>(j / 2) / degree
                          : 1 - static_cast<double>((j - 1) / 2) / degree;
          ++n;
        }
      }
      return pts;
    }
    case cell::type::hexahedron:
    {
      impl::mdarray2_t pts((degree + 3) * (degree + 2) * (degree + 1) / 6, 3);
      std::size_t n = 0;
      for (int k = 0; k <= degree; ++k)
      {
        for (int j = 0; j <= degree - k; ++j)
        {
          const auto [data, shape] = lattice::create(
              cell::type::interval, degree - j - k, latticetype, true);
          auto interval_pts = to_mdspan(data, shape);
          for (int i = 0; i <= degree - j - k; ++i)
          {
            pts(n, 0) = interval_pts(i, 0);
            pts(n, 1)
                = degree - k == 0
                      ? 0.5
                      : (j % 2 == 0 ? static_cast<double>(j / 2) / (degree - k)
                                    : 1
                                          - static_cast<double>((j - 1) / 2)
                                                / (degree - k));
            pts(n, 2) = k % 2 == 0
                            ? static_cast<double>(k / 2) / degree
                            : 1 - static_cast<double>((k - 1) / 2) / degree;
            ++n;
          }
        }
      }
      return pts;
    }
    default:
      throw std::runtime_error("Invalid cell type");
    }
  }
  else if (variant == element::dpc_variant::diagonal_equispaced
           or variant == element::dpc_variant::diagonal_gll)
  {
    lattice::type latticetype;
    lattice::simplex_method latticesm = lattice::simplex_method::isaac;
    if (variant == element::dpc_variant::diagonal_equispaced)
      latticetype = lattice::type::equispaced;
    else if (variant == element::dpc_variant::diagonal_gll)
      latticetype = lattice::type::gll;

    switch (celltype)
    {
    case cell::type::quadrilateral:
    {
      impl::mdarray2_t pts((degree + 2) * (degree + 1) / 2, 2);
      const double gap = static_cast<double>(2 * (degree + 1))
                         / (degree * degree + degree + 1);

      std::size_t n = 0;
      for (int j = 0; j <= degree; ++j)
      {
        const auto [data, shape]
            = lattice::create(cell::type::interval, j, latticetype, true);
        auto interval_pts = to_mdspan(data, shape);
        const double y = gap * (j % 2 == 0 ? j / 2 : degree - (j - 1) / 2);
        const double coord0 = y < 1 ? y : y - 1;
        const double coord1 = y < 1 ? 0 : 1;
        for (int i = 0; i <= j; ++i)
        {
          const double x = interval_pts(i, 0);
          pts(n, 0) = coord0 * (1 - x) + coord1 * x;
          pts(n, 1) = coord1 * (1 - x) + coord0 * x;
          ++n;
        }
      }
      return pts;
    }
    case cell::type::hexahedron:
    {
      impl::mdarray2_t pts((degree + 3) * (degree + 2) * (degree + 1) / 6, 3);

      const double gap
          = static_cast<double>(3 * degree) / (degree * degree + 1);

      std::size_t n = 0;
      for (int k = 0; k <= degree; ++k)
      {
        const double z = gap * (k % 2 == 0 ? k / 2 : degree - (k - 1) / 2);
        const auto [data, shape] = lattice::create(
            cell::type::triangle, k, latticetype, true, latticesm);
        auto triangle_pts = to_mdspan(data, shape);
        if (z < 1)
        {
          for (std::size_t p = 0; p < triangle_pts.extent(0); ++p)
          {
            const double coord0 = triangle_pts(p, 0);
            const double coord1 = triangle_pts(p, 1);
            pts(n, 0) = coord0 * z;
            pts(n, 1) = coord1 * z;
            pts(n, 2) = (1 - coord0 - coord1) * z;
            ++n;
          }
        }
        else if (z > 2)
        {
          for (std::size_t p = 0; p < triangle_pts.extent(0); ++p)
          {
            const double coord0 = triangle_pts(p, 0);
            const double coord1 = triangle_pts(p, 1);
            pts(n, 0) = 1 - (3 - z) * coord0;
            pts(n, 1) = 1 - (3 - z) * coord1;
            pts(n, 2) = 1 - (3 - z) * (1 - coord0 - coord1);
            ++n;
          }
        }
        else
        {
          for (std::size_t p = 0; p < triangle_pts.extent(0); ++p)
          {
            const double coord0 = triangle_pts(p, 0);
            const double coord1 = triangle_pts(p, 1);
            pts(n, 0) = 1 - (2 - z) * coord0 - coord1;
            pts(n, 1) = coord0 + (z - 1) * coord1;
            pts(n, 2) = z - 1 - (z - 1) * coord0 + (2 - z) * coord1;
            ++n;
          }
        }
      }
      return pts;
    }
    default:
      throw std::runtime_error("Invalid cell type");
    }
  }
  else
    throw std::runtime_error("Unsupported_variant");
}
//----------------------------------------------------------------------------
} // namespace

//----------------------------------------------------------------------------
FiniteElement basix::element::create_serendipity(
    cell::type celltype, int degree, element::lagrange_variant lvariant,
    element::dpc_variant dvariant, bool discontinuous)
{
  if (degree == 0)
    throw std::runtime_error("Cannot create degree 0 serendipity");

  if (celltype != cell::type::interval and celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  if (lvariant == element::lagrange_variant::unset)
  {
    if (degree < 3)
      lvariant = element::lagrange_variant::equispaced;
    else
      throw std::runtime_error("serendipity elements of degree > 2 need to be "
                               "given a Lagrange variant.");
  }

  if (dvariant == element::dpc_variant::unset
      and celltype != cell::type::interval)
  {
    if (degree == 4)
      dvariant = element::dpc_variant::simplex_equispaced;
    if (degree > 4)
      throw std::runtime_error(
          "serendipity elements of degree > 4 need to be given a DPC variant.");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  // dim 0 (vertices)
  const auto [gdata, gshape] = cell::geometry(celltype);
  impl::cmdspan2_t geometry(gdata.data(), gshape);
  for (std::size_t i = 0; i < geometry.extent(0); ++i)
  {
    auto& _x = x[0].emplace_back(1, geometry.extent(1));
    for (std::size_t j = 0; j < geometry.extent(1); ++j)
      _x(0, j) = geometry(i, j);

    auto& _M = M[0].emplace_back(1, 1, 1, 1);
    _M(0, 0, 0, 0) = 1.0;
  }

  if (degree >= 2)
  {
    FiniteElement moment_space = element::create_lagrange(
        cell::type::interval, degree - 2, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
        moment_space, celltype, 1, 2 * degree - 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 1);
    x[1] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[1] = std::vector(num_ent, impl::mdarray4_t(0, 1, 0, 1));
  }

  if (tdim >= 2)
  {
    if (degree >= 4)
    {
      FiniteElement moment_space = element::create_dpc(
          cell::type::quadrilateral, degree - 4, dvariant, true);
      auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
          moment_space, celltype, 1, 2 * degree - 4);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[2].emplace_back(_x[i], xshape[0], xshape[1]);
        M[2].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
      x[2] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
      M[2] = std::vector(num_ent, impl::mdarray4_t(0, 1, 0, 1));
    }
  }

  if (tdim == 3)
  {
    if (degree >= 6)
    {
      auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
          element::create_dpc(cell::type::hexahedron, degree - 6, dvariant,
                              true),
          celltype, 1, 2 * degree - 6);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(_x[i], xshape[0], xshape[1]);
        M[3].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray4_t(0, 1, 0, 1));
    }
  }

  std::vector<double> wbuffer;
  std::array<std::size_t, 2> wshape;
  if (tdim == 1)
  {
    wbuffer = math::eye(degree + 1);
    wshape = {static_cast<std::size_t>(degree + 1),
              static_cast<std::size_t>(degree + 1)};
  }
  else if (tdim == 2)
  {
    auto w = make_serendipity_space_2d(degree);
    wbuffer.assign(w.data(), w.data() + w.size());
    wshape = {w.extent(0), w.extent(1)};
  }
  else if (tdim == 3)
  {
    auto w = make_serendipity_space_3d(degree);
    wbuffer.assign(w.data(), w.data() + w.size());
    wshape = {w.extent(0), w.extent(1)};
  }
  else
  {
    throw std::runtime_error("Unsupported tdim");
  }

  std::array<std::vector<cmdspan2_t>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<cmdspan4_t>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, 1);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  return FiniteElement(element::family::serendipity, celltype, degree, {},
                       impl::cmdspan2_t(wbuffer.data(), wshape), xview, Mview,
                       0, maps::type::identity, discontinuous,
                       degree < static_cast<int>(tdim) ? 1 : degree / tdim,
                       degree, lvariant, dvariant);
}
//----------------------------------------------------------------------------
FiniteElement basix::element::create_dpc(cell::type celltype, int degree,
                                         element::dpc_variant variant,
                                         bool discontinuous)
{
  // Only tabulate for scalar. Vector spaces can easily be built from
  // the scalar space.
  if (!discontinuous)
    throw std::runtime_error("Cannot create a continuous DPC element.");

  if (variant == element::dpc_variant::unset)
  {
    if (degree == 0)
      variant = element::dpc_variant::simplex_equispaced;
    else
    {
      throw std::runtime_error(
          "DPC elements of degree > 0 need to be given a variant.");
    }
  }

  cell::type simplex_type;
  switch (celltype)
  {
  case cell::type::quadrilateral:
    simplex_type = cell::type::triangle;
    break;
  case cell::type::hexahedron:
    simplex_type = cell::type::tetrahedron;
    break;
  default:
    throw std::runtime_error("Invalid cell type");
  }

  if (variant == element::dpc_variant::legendre)
    return create_legendre_dpc(celltype, degree, discontinuous);

  const std::size_t ndofs = polyset::dim(simplex_type, degree);
  const std::size_t psize = polyset::dim(celltype, degree);
  impl::mdarray2_t wcoeffs(ndofs, psize);
  if (celltype == cell::type::quadrilateral)
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
      for (int j = 0; j <= degree - i; ++j)
        wcoeffs(row_n++, i * (degree + 1) + j) = 1;
  }
  else
  {
    int row_n = 0;
    for (int i = 0; i <= degree; ++i)
    {
      for (int j = 0; j <= degree - i; ++j)
      {
        for (int k = 0; k <= degree - i - j; ++k)
        {
          wcoeffs(row_n++,
                  i * (degree + 1) * (degree + 1) + j * (degree + 1) + k)
              = 1;
        }
      }
    }
  }

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  const std::size_t tdim = cell::topological_dimension(celltype);
  for (std::size_t i = 0; i < tdim; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, 1, 0, 1));
  }

  auto& _M = M[tdim].emplace_back(ndofs, 1, ndofs, 1);
  for (std::size_t i = 0; i < _M.extent(0); ++i)
    _M(i, 0, i, 0) = 1.0;

  const impl::mdarray2_t pt = make_dpc_points(celltype, degree, variant);
  x[tdim].push_back(pt);

  return FiniteElement(element::family::DPC, celltype, degree, {},
                       impl::mdspan2_t(wcoeffs.data(), wcoeffs.extents()),
                       impl::to_mdspan(x), impl::to_mdspan(M), 0,
                       maps::type::identity, discontinuous, degree, degree,
                       variant);
}
//-----------------------------------------------------------------------------
FiniteElement basix::element::create_serendipity_div(
    cell::type celltype, int degree, element::lagrange_variant lvariant,
    element::dpc_variant dvariant, bool discontinuous)
{
  if (degree == 0)
    throw std::runtime_error("Cannot create degree 0 serendipity");

  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::quadrilateral;

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;

  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  {
    FiniteElement facet_moment_space
        = facettype == cell::type::interval
              ? element::create_lagrange(facettype, degree, lvariant, true)
              : element::create_dpc(facettype, degree, dvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments(
        facet_moment_space, celltype, tdim, 2 * degree + 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim - 1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2],
                               Mshape[3]);
    }
  }

  if (degree >= 2)
  {
    FiniteElement cell_moment_space
        = element::create_dpc(celltype, degree - 2, dvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
        cell_moment_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[tdim] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  std::vector<double> wbuffer;
  std::array<std::size_t, 2> wshape;
  if (tdim == 2)
  {
    auto w = make_serendipity_div_space_2d(degree);
    wbuffer.assign(w.data(), w.data() + w.size());
    wshape = {w.extent(0), w.extent(1)};
  }
  else if (tdim == 3)
  {
    auto w = make_serendipity_div_space_3d(degree);
    wbuffer.assign(w.data(), w.data() + w.size());
    wshape = {w.extent(0), w.extent(1)};
  }
  else
  {
    throw std::runtime_error("Unsupported tdim");
  }

  std::array<std::vector<cmdspan2_t>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<cmdspan4_t>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  return FiniteElement(element::family::BDM, celltype, degree, {tdim},
                       impl::cmdspan2_t(wbuffer.data(), wshape), xview, Mview,
                       0, maps::type::contravariantPiola, discontinuous,
                       degree / tdim, degree + 1, lvariant, dvariant);
}
//-----------------------------------------------------------------------------
FiniteElement basix::element::create_serendipity_curl(
    cell::type celltype, int degree, element::lagrange_variant lvariant,
    element::dpc_variant dvariant, bool discontinuous)
{
  if (degree == 0)
    throw std::runtime_error("Cannot create degree 0 serendipity");

  if (celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Invalid celltype");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_Qpts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, celltype, 2 * degree + 1);
  impl::cmdspan2_t Qpts(_Qpts.data(), wts.size(), _Qpts.size() / wts.size());

  std::vector<double> wbuffer;
  std::array<std::size_t, 2> wshape;
  if (tdim == 2)
  {
    auto w = make_serendipity_curl_space_2d(degree);
    wbuffer.assign(w.data(), w.data() + w.size());
    wshape = {w.extent(0), w.extent(1)};
  }
  else if (tdim == 3)
  {
    auto w = make_serendipity_curl_space_3d(degree);
    wbuffer.assign(w.data(), w.data() + w.size());
    wshape = {w.extent(0), w.extent(1)};
  }
  else
  {
    throw std::runtime_error("Unsupported tdim");
  }

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 0);
    x[0] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[0] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  {
    FiniteElement edge_moment_space = element::create_lagrange(
        cell::type::interval, degree, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments(
        edge_moment_space, celltype, tdim, 2 * degree + 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }

  if (degree >= 2)
  {
    // Face integral moment
    FiniteElement moment_space = element::create_dpc(
        cell::type::quadrilateral, degree - 2, dvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
        moment_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[2].emplace_back(_x[i], xshape[0], xshape[1]);
      M[2].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[2] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  if (tdim == 3)
  {
    if (degree >= 4)
    {
      // Interior integral moment
      auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
          element::create_dpc(cell::type::hexahedron, degree - 4, dvariant,
                              true),
          celltype, tdim, 2 * degree - 3);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(_x[i], xshape[0], xshape[1]);
        M[3].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
    }
  }

  std::array<std::vector<cmdspan2_t>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<cmdspan4_t>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  return FiniteElement(element::family::N2E, celltype, degree, {tdim},
                       impl::cmdspan2_t(wbuffer.data(), wshape), xview, Mview,
                       0, maps::type::covariantPiola, discontinuous,
                       (degree == 2 && tdim == 3) ? 1 : degree / tdim,
                       degree + 1, lvariant, dvariant);
}
//-----------------------------------------------------------------------------
