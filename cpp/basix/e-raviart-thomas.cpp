// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-raviart-thomas.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "maps.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::element::create_rt(cell::type celltype, int degree,
                                        element::lagrange_variant lvariant,
                                        bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);
  const cell::type facettype
      = (tdim == 2) ? cell::type::interval : cell::type::triangle;

  // The number of order (degree-1) scalar polynomials
  const std::size_t nv = polyset::dim(celltype, degree - 1);

  // The number of order (degree-2) scalar polynomials
  const std::size_t ns0 = polyset::dim(celltype, degree - 2);

  // The number of additional polynomials in the polynomial basis for
  // Raviart-Thomas
  const std::size_t ns = polyset::dim(facettype, degree - 1);

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature(
      quadrature::type::Default, celltype, 2 * degree);
  impl::cmdspan2_t pts(_pts.data(), wts.size(), _pts.size() / wts.size());
  const auto [_phi, shape] = polyset::tabulate(celltype, degree, 0, pts);
  impl::cmdspan3_t phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const std::size_t psize = phi.extent(1);

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray2_t B(nv * tdim + ns, psize * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      B(nv * i + j, psize * i + j) = 1.0;

  // Create coefficients for additional polynomials in Raviart-Thomas
  // polynomial basis
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t k = 0; k < psize; ++k)
    {
      for (std::size_t j = 0; j < tdim; ++j)
      {
        B(nv * tdim + i, k + psize * j) = 0.0;
        for (std::size_t k1 = 0; k1 < wts.size(); ++k1)
        {
          B(nv * tdim + i, k + psize * j)
              += wts[k1] * phi(0, ns0 + i, k1) * pts(k1, j) * phi(0, k, k1);
        }
      }
    }
  }

  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;
  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  // Add integral moments on facets
  {
    const FiniteElement facet_moment_space
        = element::create_lagrange(facettype, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments(
        facet_moment_space, celltype, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim - 1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2],
                               Mshape[3]);
    }
  }

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments(
        element::create_lagrange(celltype, degree - 2, lvariant, true),
        celltype, tdim, 2 * degree - 2);
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

  return FiniteElement(element::family::RT, celltype, degree, {tdim},
                       impl::mdspan2_t(B.data(), B.extents()), xview, Mview, 0,
                       maps::type::contravariantPiola, sobolev::space::HDiv,
                       discontinuous, degree - 1, degree, lvariant);
}
//-----------------------------------------------------------------------------
