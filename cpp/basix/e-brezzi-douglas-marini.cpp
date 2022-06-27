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
#include <vector>

using namespace basix;
namespace stdex = std::experimental;

namespace
{
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;
using mdarray4_t = stdex::mdarray<double, stdex::dextents<std::size_t, 4>>;

std::array<std::vector<cmdspan2_t>, 4>
to_mdspan(std::array<std::vector<mdarray2_t>, 4>& x)
{
  std::array<std::vector<cmdspan2_t>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].emplace_back(x[i][j].data(), x[i][j].extents());

  return x1;
}
//----------------------------------------------------------------------------
std::array<std::vector<cmdspan4_t>, 4>
to_mdspan(std::array<std::vector<mdarray4_t>, 4>& M)
{
  std::array<std::vector<cmdspan4_t>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].emplace_back(M[i][j].data(), M[i][j].extents());

  return M1;
}
//----------------------------------------------------------------------------
std::array<
    std::vector<stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>>,
    4>
to_mdspan(const std::array<std::vector<std::vector<double>>, 4>& x,
          const std::array<std::vector<std::array<std::size_t, 2>>, 4>& shape)
{
  std::array<
      std::vector<stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>>,
      4>
      x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].push_back(cmdspan2_t(x[i][j].data(), shape[i][j]));

  return x1;
}
//----------------------------------------------------------------------------
std::array<
    std::vector<stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>>,
    4>
to_mdspan(const std::array<std::vector<std::vector<double>>, 4>& M,
          const std::array<std::vector<std::array<std::size_t, 4>>, 4>& shape)
{
  std::array<
      std::vector<stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>>,
      4>
      M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].push_back(cmdspan4_t(M[i][j].data(), shape[i][j]));

  return M1;
}
//----------------------------------------------------------------------------

} // namespace

//----------------------------------------------------------------------------
FiniteElement element::create_bdm(cell::type celltype, int degree,
                                  lagrange_variant lvariant, bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Unsupported cell type");

  const std::size_t tdim = cell::topological_dimension(celltype);
  std::array<std::vector<mdarray2_t>, 4> x;
  std::array<std::vector<mdarray4_t>, 4> M;
  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, i);
    x[i] = std::vector<mdarray2_t>(num_ent, mdarray2_t(0, tdim));
    M[i] = std::vector<mdarray4_t>(num_ent, mdarray4_t(0, tdim, 0, 1));
  }

  // Integral moments on facets
  const cell::type facettype = sub_entity_type(celltype, tdim - 1, 0);
  const FiniteElement facet_moment_space
      = create_lagrange(facettype, degree, lvariant, true);
  {
    auto [_x, xshape, _M, Mshape] = moments::make_normal_integral_moments_new(
        facet_moment_space, celltype, tdim, degree * 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim - 1].emplace_back(_x[i], xshape[i][0], xshape[i][1]);
      M[tdim - 1].emplace_back(_M[i], Mshape[i][0], Mshape[i][1], Mshape[i][2],
                               Mshape[i][3]);
    }
  }

  // Integral moments on interior
  if (degree > 1)
  {
    // Interior integral moment
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments_new(
        create_nedelec(celltype, degree - 1, lvariant, true), celltype, tdim,
        2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[tdim].emplace_back(_x[i], xshape[i][0], xshape[i][1]);
      M[tdim].emplace_back(_M[i], Mshape[i][0], Mshape[i][1], Mshape[i][2],
                           Mshape[i][3]);
    }
  }
  else
  {
    std::size_t num_ent = cell::num_sub_entities(celltype, tdim);
    x[tdim] = std::vector<mdarray2_t>(num_ent, mdarray2_t(0, tdim));
    M[tdim] = std::vector<mdarray4_t>(num_ent, mdarray4_t(0, tdim, 0, 1));
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  std::array<std::vector<mdspan2_t>, 4> xview = to_mdspan(x);
  std::array<std::vector<mdspan4_t>, 4> Mview = to_mdspan(M);
  std::array<std::vector<std::vector<double>>, 4> xbuffer;
  std::array<std::vector<std::vector<double>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = make_discontinuous(xview, Mview, tdim, tdim);
    xview = to_mdspan(xbuffer, xshape);
    Mview = to_mdspan(Mbuffer, Mshape);
  }

  // The number of order (degree) scalar polynomials
  const std::size_t ndofs = tdim * polyset::dim(celltype, degree);

  return FiniteElement(family::BDM, celltype, degree, {tdim},
                       mdspan2_t(math::eye(ndofs).data(), ndofs, ndofs), xview,
                       Mview, 0, maps::type::contravariantPiola, discontinuous,
                       degree, degree, lvariant);
}
//-----------------------------------------------------------------------------
