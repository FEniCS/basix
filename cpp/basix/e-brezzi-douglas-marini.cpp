// Copyright (c) 2020 Chris Richardson and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-brezzi-douglas-marini.h"
#include "e-lagrange.h"
#include "e-nedelec.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polyset.h"
#include "sobolev-spaces.h"
#include <vector>

using namespace basix;

//----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> element::create_bdm(cell::type celltype, int degree,
                                     lagrange_variant lvariant,
                                     bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);
  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  // Integral moments on facets
  const cell::type facettype = sub_entity_type(celltype, tdim - 1, 0);
  const FiniteElement<T> facet_moment_space
      = create_lagrange<T>(facettype, degree, lvariant, true);
  {
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments<T>(
        facet_moment_space, celltype, polyset::type::standard, tdim,
        degree * 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[tdim - 1].emplace_back(Mshape, _M[i]);
    }
  }

  // Integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<T>(
        create_nedelec<T>(celltype, degree - 1, lvariant, true), celltype,
        polyset::type::standard, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[tdim].emplace_back(Mshape, _M[i]);
    }
  }
  else
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[tdim] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  // The number of order (degree) scalar polynomials
  const std::size_t ndofs
      = tdim * polyset::dim(celltype, polyset::type::standard, degree);
  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HDiv;
  return FiniteElement<T>(
      family::BDM, celltype, polyset::type::standard, degree, {tdim},
      impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs), xview,
      Mview, 0, maps::type::contravariantPiola, space, discontinuous, degree,
      degree, lvariant, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template FiniteElement<float> element::create_bdm(cell::type, int,
                                                  lagrange_variant, bool);
template FiniteElement<double> element::create_bdm(cell::type, int,
                                                   lagrange_variant, bool);
//-----------------------------------------------------------------------------
