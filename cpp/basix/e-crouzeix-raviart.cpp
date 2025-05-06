// Copyright (c) 2020-2025 Chris Richardson, Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-crouzeix-raviart.h"
#include "cell.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <array>
#include <vector>

using namespace basix;

//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> basix::element::create_cr(cell::type celltype, int degree,
                                           bool discontinuous)
{
  if (degree != 1)
    throw std::runtime_error("Degree must be 1 for Crouzeix-Raviart");

  const std::size_t tdim = cell::topological_dimension(celltype);
  if (tdim < 2)
  {
    throw std::runtime_error(
        "topological dim must be 2 or 3 for Crouzeix-Raviart");
  }

  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron
      and celltype != cell::type::quadrilateral
      and celltype != cell::type::hexahedron)
  {
    throw std::runtime_error("Crouzeix-Raviart is only defined on triangles, "
                             "quadrilaterals, tetrahedra and hexahedra.");
  }

  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);
  const std::vector<std::vector<int>>& facet_topology = topology[tdim - 1];
  const std::size_t ndofs = facet_topology.size();

  const auto [gdata, shape] = cell::geometry<T>(celltype);
  impl::mdspan_t<const T, 2> geometry(gdata.data(), shape);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;
  for (std::size_t i = 0; i < tdim - 1; ++i)
  {
    const std::size_t num_ent = topology[i].size();
    x[i] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[i] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, 1, 0, 1));
  }

  x[tdim - 1]
      = std::vector(facet_topology.size(), impl::mdarray_t<T, 2>(1, tdim));
  M[tdim - 1] = std::vector(
      facet_topology.size(),
      impl::mdarray_t<T, 4>(std::array<std::size_t, 4>{1, 1, 1, 1}, 1));

  // Compute facet midpoints
  for (std::size_t f = 0; f < facet_topology.size(); ++f)
  {
    const std::vector<int>& ft = facet_topology[f];
    auto& _x = x[tdim - 1][f];
    for (std::size_t i = 0; i < ft.size(); ++i)
    {
      for (std::size_t j = 0; j < geometry.extent(1); ++j)
        _x(0, j) += geometry(ft[i], j) / ft.size();
    }
  }

  x[tdim] = std::vector(topology[tdim].size(), impl::mdarray_t<T, 2>(0, tdim));
  M[tdim]
      = std::vector(topology[tdim].size(), impl::mdarray_t<T, 4>(0, 1, 0, 1));
  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, 1);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  if (celltype == cell::type::triangle or celltype == cell::type::tetrahedron)
  {
    return FiniteElement<T>(
        element::family::CR, celltype, polyset::type::standard, 1, {},
        impl::mdspan_t<T, 2>(math::eye<T>(ndofs).data(), ndofs, ndofs), xview,
        Mview, 0, maps::type::identity, sobolev::space::L2, discontinuous,
        degree, degree, element::lagrange_variant::unset,
        element::dpc_variant::unset);
  }
  else if (celltype == cell::type::quadrilateral)
  {
    const auto [_pts, wts] = quadrature::make_quadrature<T>(
        quadrature::type::Default, cell::type::quadrilateral,
        polyset::type::standard, 6);
    impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                   _pts.size() / wts.size());

    const auto [_phi, shape] = polyset::tabulate(
        cell::type::quadrilateral, polyset::type::standard, degree + 1, 0, pts);
    impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

    impl::mdarray_t<T, 2> wcoeffs(ndofs, 9);
    wcoeffs(0, 0) = 1;
    wcoeffs(1, 1) = 1;
    wcoeffs(2, 3) = 1;
    for (int i = 2; i < 9; ++i)
    {
      if (i != 3)
      {
        wcoeffs(3, i) = 0.0;
        for (std::size_t k = 0; k < wts.size(); ++k)
          wcoeffs(3, i) += wts[k] * (pts(k, 0) + pts(k, 1))
                           * (pts(k, 0) - pts(k, 1)) * phi(0, i, k);
      }
    }

    math::orthogonalise<T>(wcoeffs, 3);

    return FiniteElement<T>(
        element::family::CR, celltype, polyset::type::standard, 1, {}, wcoeffs,
        xview, Mview, 0, maps::type::identity, sobolev::space::L2,
        discontinuous, 0, degree + 1, element::lagrange_variant::unset,
        element::dpc_variant::unset);
  }
  else if (celltype == cell::type::hexahedron)
  {
    const auto [_pts, wts] = quadrature::make_quadrature<T>(
        quadrature::type::Default, cell::type::hexahedron,
        polyset::type::standard, 6);
    impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                   _pts.size() / wts.size());

    const auto [_phi, shape] = polyset::tabulate(
        cell::type::hexahedron, polyset::type::standard, degree + 1, 0, pts);
    impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

    impl::mdarray_t<T, 2> wcoeffs(ndofs, 27);
    wcoeffs(0, 0) = 1;
    wcoeffs(1, 1) = 1;
    wcoeffs(2, 3) = 1;
    wcoeffs(3, 9) = 1;
    for (int i = 2; i < 27; ++i)
    {
      if (i != 3 and i != 9)
      {
        wcoeffs(4, i) = 0.0;
        wcoeffs(5, i) = 0.0;
        for (std::size_t k = 0; k < wts.size(); ++k)
        {
          wcoeffs(4, i) += wts[k] * (pts(k, 0) + pts(k, 1))
                           * (pts(k, 0) - pts(k, 1)) * phi(0, i, k);
          wcoeffs(5, i) += wts[k] * (pts(k, 0) + pts(k, 2))
                           * (pts(k, 0) - pts(k, 2)) * phi(0, i, k);
        }
      }
    }

    math::orthogonalise<T>(wcoeffs, 4);

    return FiniteElement<T>(
        element::family::CR, celltype, polyset::type::standard, 1, {}, wcoeffs,
        xview, Mview, 0, maps::type::identity, sobolev::space::L2,
        discontinuous, 0, degree + 1, element::lagrange_variant::unset,
        element::dpc_variant::unset);
  }
  else
  {
    throw std::runtime_error("Invalid cell type");
  }
}
//-----------------------------------------------------------------------------
template FiniteElement<float> element::create_cr(cell::type, int, bool);
template FiniteElement<double> element::create_cr(cell::type, int, bool);
//-----------------------------------------------------------------------------
