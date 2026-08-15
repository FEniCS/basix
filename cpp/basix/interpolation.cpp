// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "interpolation.h"
#include "finite-element.h"
#include "math.h"
#include <concepts>
#include <exception>

using namespace basix;

template <typename T, std::size_t D>
using mdspan_t = md::mdspan<T, md::dextents<std::size_t, D>>;

//----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
basix::compute_interpolation_operator(const FiniteElement<T>& element_from,
                                      const FiniteElement<T>& element_to)
{
  if (element_from.cell_type() != element_to.cell_type())
  {
    throw std::runtime_error(
        "Cannot interpolate between elements defined on different cell types.");
  }

  const auto [points, pts_shape] = element_to.points();
  const auto [tab_b, tab_shape] = element_from.tabulate(
      0, mdspan_t<const T, 2>(points.data(), pts_shape));
  mdspan_t<const T, 4> tab(tab_b.data(), tab_shape);
  const auto [imb, imshape] = element_to.interpolation_matrix();
  mdspan_t<const T, 2> i_m(imb.data(), imshape);

  const std::size_t dim_to = element_to.dim();
  const std::size_t dim_from = element_from.dim();
  const std::size_t npts = tab.extent(1);

  const std::size_t vs_from = std::accumulate(
      element_from.value_shape().begin(), element_from.value_shape().end(),
      std::size_t{1}, std::multiplies{});
  const std::size_t vs_to = std::reduce(element_to.value_shape().begin(),
                                        element_to.value_shape().end(),
                                        std::size_t{1}, std::multiplies{});

  if (vs_from != vs_to)
  {
    if (vs_to == 1)
    {
      // Map element_from's components into element_to
      std::array<std::size_t, 2> shape = {dim_to * vs_from, dim_from};
      std::vector<T> outb(shape[0] * shape[1], 0.0);
      mdspan_t<T, 2> out(outb.data(), shape);
      for (std::size_t i = 0; i < vs_from; ++i)
        for (std::size_t j = 0; j < dim_to; ++j)
          for (std::size_t k = 0; k < dim_from; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out[i + j * vs_from, k] += i_m[j, l] * tab[0, l, k, i];

      return {std::move(outb), shape};
    }
    else if (vs_from == 1)
    {
      // Map duplicates of element_to to components of element_to
      std::array<std::size_t, 2> shape = {dim_to, dim_from * vs_to};
      std::vector<T> outb(shape[0] * shape[1], 0.0);
      mdspan_t<T, 2> out(outb.data(), shape);
      for (std::size_t i = 0; i < vs_to; ++i)
        for (std::size_t j = 0; j < dim_from; ++j)
          for (std::size_t k = 0; k < dim_to; ++k)
            for (std::size_t l = 0; l < npts; ++l)
              out[k, i + j * vs_to] += i_m[k, i * npts + l] * tab[0, l, j, 0];

      return {std::move(outb), shape};
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
    std::vector<T> outb(shape[0] * shape[1], 0.0);
    mdspan_t<T, 2> out(outb.data(), shape);

    // i_m's columns are ordered as (k, l) -> k * npts + l for the first
    // vs_from * npts columns, matching tab's layout below. i_m can have
    // *more* columns than vs_from * npts (e.g. elements with
    // interpolation_nderivs > 0, such as Hermite, encode extra derivative
    // moments there); only the first vs_from * npts are used here, exactly
    // as the quadruple loop this replaces only ever indexed columns in
    // that range. Since those columns are not necessarily i_m's full row
    // length, they must be copied into a compact buffer rather than
    // viewed in-place, so that the matrix-matrix product below (computed
    // with BLAS via math::dot) reads/writes only within bounds.
    const std::size_t ncols = vs_from * npts;
    std::vector<T> Ab(dim_to * ncols);
    mdspan_t<T, 2> A(Ab.data(), std::array<std::size_t, 2>{dim_to, ncols});
    for (std::size_t i = 0; i < dim_to; ++i)
      for (std::size_t c = 0; c < ncols; ++c)
        A[i, c] = i_m[i, c];

    std::vector<T> Bb(ncols * dim_from);
    mdspan_t<T, 2> B(Bb.data(), std::array<std::size_t, 2>{ncols, dim_from});
    for (std::size_t k = 0; k < vs_from; ++k)
      for (std::size_t l = 0; l < npts; ++l)
        for (std::size_t j = 0; j < dim_from; ++j)
          B[k * npts + l, j] = tab[0, l, j, k];

    math::dot(A, B, out);

    return {std::move(outb), shape};
  }
}
//----------------------------------------------------------------------------
/// @cond DOXYGEN_SHOULD_SKIP_THIS
template std::pair<std::vector<float>, std::array<std::size_t, 2>>
basix::compute_interpolation_operator(const FiniteElement<float>&,
                                      const FiniteElement<float>&);
template std::pair<std::vector<double>, std::array<std::size_t, 2>>
basix::compute_interpolation_operator(const FiniteElement<double>&,
                                      const FiniteElement<double>&);
/// @endcond
//-----------------------------------------------------------------------------
