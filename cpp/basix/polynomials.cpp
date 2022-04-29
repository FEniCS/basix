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
                                             const xt::xtensor<double, 2>& x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
    return xt::view(polyset::tabulate(celltype, d, 0, x), 0, xt::all(),
                    xt::all());
  default:
    throw std::runtime_error("not implemented yet");
  }
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type, cell::type cell, int d)
{
  return polyset::dim(cell, d);
}
