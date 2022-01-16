// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "interpolation.h"
#include "finite-element.h"
#include <exception>
#include <xtensor/xtensor.hpp>

using namespace basix;

//----------------------------------------------------------------------------
xt::xtensor<double, 2>
basix::compute_interpolation_operator(const FiniteElement& element_from,
                                      const FiniteElement& element_to)
{
  if (element_from.cell_type() != element_to.cell_type())
  {
    throw std::runtime_error(
        "Cannot interpolate between elements defined on different cell types.");
  }

  const xt::xtensor<double, 2>& points = element_to.points();
  const xt::xtensor<double, 4> tab = element_from.tabulate(0, points);
  const xt::xtensor<double, 2>& i_m = element_to.interpolation_matrix();

  const std::size_t dim_to = element_to.dim();
  const std::size_t dim_from = element_from.dim();
  const std::size_t npts = tab.shape(1);

  const std::size_t vs_from = std::accumulate(
      element_from.value_shape().begin(), element_from.value_shape().end(), 1,
      std::multiplies<int>());
  const std::size_t vs_to = std::accumulate(element_to.value_shape().begin(),
                                            element_to.value_shape().end(), 1,
                                            std::multiplies<int>());

  if (vs_from != vs_to)
  {
    if (vs_to == 1)
    {
      // Map element_from's components into element_to
      xt::xtensor<double, 2> out({dim_to * vs_from, dim_from});
      out.fill(0);
      for (std::size_t i = 0; i < vs_from; ++i)
        for (std::size_t j = 0; j < dim_to; ++j)
          for (std::size_t k = 0; k < dim_from; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out(i + j * vs_from, k) += i_m(j, l) * tab(0, l, k, i);

      return out;
    }
    else if (vs_from == 1)
    {
      // Map duplicates of element_to to components of element_to
      xt::xtensor<double, 2> out({dim_to, dim_from * vs_to});
      out.fill(0);
      for (std::size_t i = 0; i < vs_to; ++i)
        for (std::size_t j = 0; j < dim_from; ++j)
          for (std::size_t k = 0; k < dim_to; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out(k, i + j * vs_to) += i_m(k, i * npts + l) * tab(0, l, j, 0);

      return out;
    }
    else
    {
      throw std::runtime_error("Cannot interpolate between elements with this "
                               "combination of value sizes.");
    }
  }
  else
  {
    xt::xtensor<double, 2> out({dim_to, dim_from});
    out.fill(0);
    for (std::size_t i = 0; i < dim_to; ++i)
      for (std::size_t j = 0; j < dim_from; ++j)
        for (std::size_t k = 0; k < vs_from; ++k)
          for (std::size_t l = 0; l < npts; ++l)
            out(i, j) += i_m(i, k * npts + l) * tab(0, l, j, k);

    return out;
  }
}
//----------------------------------------------------------------------------
