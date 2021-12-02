// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "polyset.h"

using namespace basix;

//-----------------------------------------------------------------------------
xt::xtensor<double, 2> polynomials::tabulate(polynomials::type polytype,
                                             cell::type celltype, int d,
                                             const xt::xarray<double>& x)
{
  throw std::runtime_error("not implemented yet");
  assert(polytype);
  assert(celltype);
  assert(d);
  assert(x(0, 0));
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type polytype, cell::type cell, int d)
{
  throw std::runtime_error("not implemented yet");
  assert(polytype);
  assert(cell);
  assert(d);
}
