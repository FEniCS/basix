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
    return xt::view(polyset::tabulate(celltype, d, 0, x), 0, xt::all(),
                    xt::all());
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
