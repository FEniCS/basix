// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "interpolation.h"
#include <exception>
#include <xtensor/xtensor.hpp>

using namespace basix;

//----------------------------------------------------------------------------
xt::xtensor<double, 2>
basix::compute_interpolation_between_elements(const FiniteElement& element_from,
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

  if (element_from.value_size() != element_to.value_size())
  {
    if (element_to.value_size() == 1)
    {
      // Map element_from's components into element_to
      const std::size_t vs = element_from.value_size();
      xt::xtensor<double, 2> out({dim_to * vs, dim_from});
      out.fill(0);
      for (std::size_t i = 0; i < vs; ++i)
        for (std::size_t j = 0; j < dim_to; ++j)
          for (std::size_t k = 0; k < dim_from; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out(i + j * vs, k) += i_m(j, l) * tab(0, l, k, i);

      return out;
    }
    else if (element_from.value_size() == 1)
    {
      // Map duplicates of element_to to components of element_to
      const std::size_t vs = element_to.value_size();
      xt::xtensor<double, 2> out({dim_to, dim_from * vs});
      out.fill(0);
      for (std::size_t i = 0; i < vs; ++i)
        for (std::size_t j = 0; j < dim_from; ++j)
          for (std::size_t k = 0; k < dim_to; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out(k, i + j * vs) += i_m(k, i * npts + l) * tab(0, l, j, 0);

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
    const std::size_t vs = element_from.value_size();
    xt::xtensor<double, 2> out({dim_to, dim_from});
    out.fill(0);
    for (std::size_t i = 0; i < dim_to; ++i)
      for (std::size_t j = 0; j < dim_from; ++j)
        for (std::size_t k = 0; k < vs; ++k)
          for (std::size_t l = 0; l < npts; ++l)
            out(i, j) += i_m(i, k * npts + l) * tab(0, l, j, k);

    return out;
  }
}
//----------------------------------------------------------------------------
