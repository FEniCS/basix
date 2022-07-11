// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "interpolation.h"
#include "finite-element.h"
#include <exception>

using namespace basix;

namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

//----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
basix::compute_interpolation_operator(const FiniteElement& element_from,
                                      const FiniteElement& element_to)
{
  if (element_from.cell_type() != element_to.cell_type())
  {
    throw std::runtime_error(
        "Cannot interpolate between elements defined on different cell types.");
  }

  const auto [points, shape] = element_to.points();
  const auto [tab_b, tab_shape]
      = element_from.tabulate(0, cmdspan2_t(points.data(), shape));
  cmdspan4_t tab(tab_b.data(), tab_shape);
  const auto [imb, imshape] = element_to.interpolation_matrix();
  cmdspan2_t i_m(imb.data(), imshape);

  const std::size_t dim_to = element_to.dim();
  const std::size_t dim_from = element_from.dim();
  const std::size_t npts = tab.extent(1);

  const std::size_t vs_from
      = std::accumulate(element_from.value_shape().begin(),
                        element_from.value_shape().end(), 1, std::multiplies{});
  const std::size_t vs_to
      = std::reduce(element_to.value_shape().begin(),
                    element_to.value_shape().end(), 1, std::multiplies{});

  if (vs_from != vs_to)
  {
    if (vs_to == 1)
    {
      // Map element_from's components into element_to
      std::array<std::size_t, 2> shape = {dim_to * vs_from, dim_from};
      std::vector<double> outb(shape[0] * shape[1], 0.0);
      mdspan2_t out(outb.data(), shape);
      for (std::size_t i = 0; i < vs_from; ++i)
        for (std::size_t j = 0; j < dim_to; ++j)
          for (std::size_t k = 0; k < dim_from; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out(i + j * vs_from, k) += i_m(j, l) * tab(0, l, k, i);

      return {std::move(outb), std::move(shape)};
    }
    else if (vs_from == 1)
    {
      // Map duplicates of element_to to components of element_to
      std::array<std::size_t, 2> shape = {dim_to, dim_from * vs_to};
      std::vector<double> outb(shape[0] * shape[1], 0.0);
      mdspan2_t out(outb.data(), shape);
      for (std::size_t i = 0; i < vs_to; ++i)
        for (std::size_t j = 0; j < dim_from; ++j)
          for (std::size_t k = 0; k < dim_to; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out(k, i + j * vs_to) += i_m(k, i * npts + l) * tab(0, l, j, 0);

      return {std::move(outb), std::move(shape)};
    }
    else
    {
      throw std::runtime_error("Cannot interpolate between elements with this "
                               "combination of value sizes.");
    }
  }
  else
  {
    std::array<std::size_t, 2> shape = {dim_to, dim_from};
    std::vector<double> outb(shape[0] * shape[1], 0.0);
    mdspan2_t out(outb.data(), shape);
    for (std::size_t i = 0; i < dim_to; ++i)
      for (std::size_t j = 0; j < dim_from; ++j)
        for (std::size_t k = 0; k < vs_from; ++k)
          for (std::size_t l = 0; l < npts; ++l)
            out(i, j) += i_m(i, k * npts + l) * tab(0, l, j, k);

    return {std::move(outb), std::move(shape)};
  }
}
//----------------------------------------------------------------------------
