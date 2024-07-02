// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-raviart-thomas.h"
#include "e-lagrange.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <cmath>
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_rt(cell::type celltype, int degree,
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
  const std::size_t nv
      = polyset::dim(celltype, polyset::type::standard, degree - 1);

  // The number of order (degree-2) scalar polynomials
  const std::size_t ns0
      = polyset::dim(celltype, polyset::type::standard, degree - 2);

  // The number of additional polynomials in the polynomial basis for
  // Raviart-Thomas
  const std::size_t ns
      = polyset::dim(facettype, polyset::type::standard, degree - 1);

  // Evaluate the expansion polynomials at the quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, celltype, polyset::type::standard, 2 * degree);
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                 _pts.size() / wts.size());
  const auto [_phi, shape]
      = polyset::tabulate(celltype, polyset::type::standard, degree, 0, pts);
  impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

  // The number of order (degree) polynomials
  const std::size_t psize = phi.extent(1);

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray_t<T, 2> B(nv * tdim + ns, psize * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      B[nv * i + j, psize * i + j] = 1.0;

  // Create coefficients for additional polynomials in Raviart-Thomas
  // polynomial basis
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t k = nv; k < psize; ++k)
    {
      for (std::size_t j = 0; j < tdim; ++j)
      {
        B[nv * tdim + i, k + psize * j] = 0.0;
        for (std::size_t k1 = 0; k1 < wts.size(); ++k1)
        {
          B[nv * tdim + i, k + psize * j]
              += wts[k1] * phi[0, ns0 + i, k1] * pts[k1, j] * phi[0, k, k1];
        }
      }
    }
  }

  math::orthogonalise<T>(B, nv * tdim);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  // Add integral moments on facets
  {
    const FiniteElement facet_moment_space
        = element::create_lagrange<T>(facettype, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments<T>(
        facet_moment_space, celltype, polyset::type::standard, tdim,
        2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(xshape, _x[i]);
      M[tdim - 1].emplace_back(Mshape, _M[i]);
    }
  }

  // Add integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments<T>(
        element::create_lagrange<T>(celltype, degree - 2, lvariant, true),
        celltype, polyset::type::standard, tdim, 2 * degree - 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(xshape, _x[i]);
      M[tdim].emplace_back(Mshape, _M[i]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[tdim] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HDiv;
  return FiniteElement<T>(
      element::family::RT, celltype, polyset::type::standard, degree, {tdim},
      impl::mdspan_t<T, 2>(B.data(), B.extents()), xview, Mview, 0,
      maps::type::contravariantPiola, space, discontinuous, degree - 1, degree,
      lvariant, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float>
element::create_rt(cell::type, int, element::lagrange_variant, bool);
template FiniteElement<double>
element::create_rt(cell::type, int, element::lagrange_variant, bool);
//-----------------------------------------------------------------------------
