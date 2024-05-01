// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element-utils.h"
#include "dof-transformations.h"
#include "e-brezzi-douglas-marini.h"
#include "e-bubble.h"
#include "e-crouzeix-raviart.h"
#include "e-hermite.h"
#include "e-hhj.h"
#include "e-lagrange.h"
#include "e-nce-rtc.h"
#include "e-nedelec.h"
#include "e-raviart-thomas.h"
#include "e-regge.h"
#include "e-serendipity.h"
#include "math.h"
#include "polyset.h"
#include <basix/version.h>
#include <cmath>
#include <concepts>
#include <limits>
#include <numeric>
#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;

//-----------------------------------------------------------------------------
template <std::floating_point T>
std::tuple<std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 2>>, 4>,
           std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 4>>, 4>>
element::make_discontinuous(
    const std::array<std::vector<mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<mdspan_t<const T, 4>>, 4>& M, std::size_t tdim,
    std::size_t value_size)
{
  std::size_t npoints = 0;
  std::size_t Mshape0 = 0;
  for (int i = 0; i < 4; ++i)
  {
    for (std::size_t j = 0; j < x[i].size(); ++j)
    {
      npoints += x[i][j].extent(0);
      Mshape0 += M[i][j].extent(0);
    }
  }
  const std::size_t nderivs = M[0][0].extent(3);

  std::array<std::vector<std::vector<T>>, 4> x_data;
  std::array<std::vector<std::array<std::size_t, 2>>, 4> xshapes;
  std::array<std::vector<std::vector<T>>, 4> M_data;
  std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshapes;
  for (std::size_t i = 0; i < tdim; ++i)
  {
    xshapes[i] = std::vector(x[i].size(), std::array<std::size_t, 2>{0, tdim});
    x_data[i].resize(x[i].size());
    Mshapes[i] = std::vector(
        M[i].size(), std::array<std::size_t, 4>{0, value_size, 0, nderivs});
    M_data[i].resize(M[i].size());
  }

  std::array<std::size_t, 2> xshape = {npoints, tdim};
  std::vector<T> xb(xshape[0] * xshape[1]);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      new_x(xb.data(), xshape);
  std::array<std::size_t, 4> Mshape = {Mshape0, value_size, npoints, nderivs};
  std::vector<T> Mb(Mshape[0] * Mshape[1] * Mshape[2] * Mshape[3]);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>
      new_M(Mb.data(), Mshape);
  int x_n = 0;
  int M_n = 0;
  for (int i = 0; i < 4; ++i)
  {
    for (std::size_t j = 0; j < x[i].size(); ++j)
    {
      for (std::size_t k0 = 0; k0 < x[i][j].extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < x[i][j].extent(1); ++k1)
          new_x(k0 + x_n, k1) = x[i][j](k0, k1);

      for (std::size_t k0 = 0; k0 < M[i][j].extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < M[i][j].extent(1); ++k1)
          for (std::size_t k2 = 0; k2 < M[i][j].extent(2); ++k2)
            for (std::size_t k3 = 0; k3 < M[i][j].extent(3); ++k3)
              new_M(k0 + M_n, k1, k2 + x_n, k3) = M[i][j](k0, k1, k2, k3);

      x_n += x[i][j].extent(0);
      M_n += M[i][j].extent(0);
    }
  }

  x_data[tdim].push_back(xb);
  xshapes[tdim].push_back(xshape);
  M_data[tdim].push_back(Mb);
  Mshapes[tdim].push_back(Mshape);
  return {std::move(x_data), std::move(xshapes), std::move(M_data),
          std::move(Mshapes)};
}
//-----------------------------------------------------------------------------
/// @cond
template std::tuple<std::array<std::vector<std::vector<float>>, 4>,
                    std::array<std::vector<std::array<std::size_t, 2>>, 4>,
                    std::array<std::vector<std::vector<float>>, 4>,
                    std::array<std::vector<std::array<std::size_t, 4>>, 4>>
element::make_discontinuous(
    const std::array<std::vector<mdspan_t<const float, 2>>, 4>&,
    const std::array<std::vector<mdspan_t<const float, 4>>, 4>&, std::size_t,
    std::size_t);
template std::tuple<std::array<std::vector<std::vector<double>>, 4>,
                    std::array<std::vector<std::array<std::size_t, 2>>, 4>,
                    std::array<std::vector<std::vector<double>>, 4>,
                    std::array<std::vector<std::array<std::size_t, 4>>, 4>>
element::make_discontinuous(
    const std::array<std::vector<mdspan_t<const double, 2>>, 4>&,
    const std::array<std::vector<mdspan_t<const double, 4>>, 4>&, std::size_t,
    std::size_t);
/// @endcond
//-----------------------------------------------------------------------------
