// Copyright (c) 2021 Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

namespace basix::linalg
{

template <class U, class V>
auto dot(const U& u, const V& v)
{
  using value_type = std::common_type_t<typename U::storage_type::value_type,
                                        typename V::storage_type::value_type>;
  using return_type
      = std::conditional_t<(U::static_layout == V::static_layout)
                               and (U::static_layout != xt::layout_type::dynamic
                                    and U::static_layout
                                            != xt::layout_type::any),
                           xt::xarray<value_type, U::static_layout>,
                           xt::xarray<value_type, XTENSOR_DEFAULT_LAYOUT>>;
  return_type result;

  // is one of each a scalar? just multiply
  if (u.dimension() == 0 or v.dimension() == 0)
  {
    return return_type(u * v);
  }
  else if (u.dimension() == 1 and v.dimension() == 1)
  {
    // Dot product
    return return_type(u * v);
  }
  else if (u.dimension() == 2 and v.dimension() == 1)
  {
    // Matrix-vector product
    result.resize({u.shape()[0]});
    result = 0.0;
    for (std::size_t i = 0; i < u.shape()[0]; ++i)
      for (std::size_t j = 0; j < u.shape()[1]; ++j)
        result[i] += u(i, j) * v[j];
    return result;
  }
  else if (u.dimension() == 2 and v.dimension() == 2)
  {
    // Matrix-matrix product
    if (u.shape()[1] != v.shape()[0])
      throw std::runtime_error("Size-mismatch");
    result.resize({u.shape()[0], v.shape()[1]});
    result = 0.0;
    for (std::size_t i = 0; i < u.shape()[0]; ++i)
      for (std::size_t j = 0; j < u.shape()[1]; ++j)
        for (std::size_t k = 0; k < u.shape()[1]; ++k)
          result(i, j) += u(i, k) * v(k, j);
    return result;
  }
  else
    throw std::runtime_error("Unsupported case yet");
}

} // namespace basix::linalg
