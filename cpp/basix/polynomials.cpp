// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "polyset.h"
#include <xtensor/xview.hpp>

using namespace basix;

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> polynomials::tabulate(polynomials::type polytype,
                                             cell::type celltype, int d,
                                             const xt::xarray<double>& x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
  {
    xt::xtensor<double, 2> tab = xt::view(polyset::tabulate(celltype, d, 0, x),
                                          0, xt::all(), xt::all());
    switch (celltype)
    {
    case cell::type::interval:
      return tab * std::sqrt(2.0);
    case cell::type::triangle:
      return tab * 2;
    case cell::type::quadrilateral:
      return tab * 2;
    case cell::type::tetrahedron:
      return tab * 2 * std::sqrt(2);
    case cell::type::hexahedron:
      return tab * 2 * std::sqrt(2);
    default:
      return tab;
    }
  }
  default:
    throw std::runtime_error("not implemented yet a");
  }
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type polytype, cell::type cell, int d)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
    return polyset::dim(cell, d);
  default:
    throw std::runtime_error("not implemented yet a");
  }
}
