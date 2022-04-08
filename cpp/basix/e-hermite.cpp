// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-hermite.h"
#include "element-families.h"
#include "lattice.h"
#include "maps.h"
#include "polyset.h"
#include "quadrature.h"
#include <array>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

//----------------------------------------------------------------------------
FiniteElement basix::element::create_hermite(cell::type celltype, int degree,
                                             bool discontinuous)
{
  switch (celltype)
  {
  case cell::type::triangle:
    if (degree < 3)
      throw std::runtime_error(
          "Hermite element on a triangle must have degree 3");
    break;
  default:
    throw std::runtime_error("Unsupported cell type");
  }

  const std::size_t tdim = cell::topological_dimension(celltype);
  const std::size_t ndofs = polyset::dim(celltype, degree);
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(celltype);

  const std::size_t deriv_count = polyset::nderivs(celltype, 1);

  std::array<std::vector<xt::xtensor<double, 4>>, 4> M;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x;

  for (std::size_t i = 0; i <= tdim; ++i)
  {
    M[i].resize(topology[i].size());
    x[i].resize(topology[i].size());
  }

  // Loop over entities of dimension 'dim'
  for (std::size_t e = 0; e < topology[0].size(); ++e)
  {
    const xt::xtensor<double, 2> entity_x
        = cell::sub_entity_geometry(celltype, 0, e);
    x[0][e] = entity_x;
    const std::array<std::size_t, 4> sh = {3, 1, 1, deriv_count};
    M[0][e] = xt::zeros<double>(sh);
    M[0][e](0, 0, 0, 0) = 1;
    M[0][e](1, 0, 0, 1) = 1;
    M[0][e](2, 0, 0, 2) = 1;
  }

  for (std::size_t e = 0; e < topology[1].size(); ++e)
  {
    x[1][e] = xt::xtensor<double, 2>({0, tdim});
    const std::array<std::size_t, 4> sh = {0, 1, 0, deriv_count};
    M[1][e] = xt::xtensor<double, 4>(sh);
  }

  x[2][0] = {{1. / 3, 1. / 3}};
  const std::array<std::size_t, 4> sh = {1, 1, 1, deriv_count};
  M[2][0] = xt::zeros<double>(sh);
  M[2][0](0, 0, 0, 0) = 1;

  xt::xtensor<double, 2> wcoeffs = xt::eye<double>({ndofs, ndofs});

  if (discontinuous)
  {
    std::tie(x, M) = element::make_discontinuous(x, M, tdim, 1);
  }

  return FiniteElement(element::family::Hermite, celltype, degree, 1, {1},
                       wcoeffs, x, M, maps::type::identity, discontinuous, -1);
}
//-----------------------------------------------------------------------------
