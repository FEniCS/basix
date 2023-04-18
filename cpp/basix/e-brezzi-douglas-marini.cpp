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
FiniteElement element::create_bdm(cell::type celltype, int degree,
                                  lagrange_variant lvariant, bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const std::size_t tdim = cell::topological_dimension(celltype);
  std::array<std::vector<impl::mdarray2_t>, 4> x;
  std::array<std::vector<impl::mdarray4_t>, 4> M;
  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  // Integral moments on facets
  const cell::type facettype = sub_entity_type(celltype, tdim - 1, 0);
  const FiniteElement facet_moment_space
      = create_lagrange(facettype, degree, lvariant, true);
  {
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments<double>(
        facet_moment_space, celltype, tdim, degree * 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim - 1].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2],
                               Mshape[3]);
    }
  }

  // Integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<double>(
        create_nedelec(celltype, degree - 1, lvariant, true), celltype, tdim,
        2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(_x[i], xshape[0], xshape[1]);
      M[tdim].emplace_back(_M[i], Mshape[0], Mshape[1], Mshape[2], Mshape[3]);
    }
  }
  else
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector(num_ent, impl::mdarray2_t(0, tdim));
    M[tdim] = std::vector(num_ent, impl::mdarray4_t(0, tdim, 0, 1));
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<mdspan2_t<const double>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan4_t<const double>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
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
  const std::size_t ndofs = tdim * polyset::dim(celltype, degree);
  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HDiv;
  return FiniteElement(family::BDM, celltype, degree, {tdim},
                       impl::mdspan2_t<double>(math::eye(ndofs).data(), ndofs, ndofs),
                       xview, Mview, 0, maps::type::contravariantPiola, space,
                       discontinuous, degree, degree, lvariant,
                       element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
