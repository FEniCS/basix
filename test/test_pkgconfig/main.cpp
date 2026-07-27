// Copyright (c) 2022 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier: MIT

#include <basix/finite-element.h>
#include <memory>

int main()
{
  auto element0 = basix::create_element<float>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  auto element1 = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset, false);

  return 0;
}
