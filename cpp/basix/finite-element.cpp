// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
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
#include <numeric>

#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;
namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

using mdarray2_t = stdex::mdarray<double, stdex::dextents<std::size_t, 2>>;
using mdarray3_t = stdex::mdarray<double, stdex::dextents<std::size_t, 3>>;

namespace
{
//----------------------------------------------------------------------------
/// This function orthogonalises and normalises the rows of a matrix in place
void orthogonalise(mdspan2_t& wcoeffs)
{
  for (std::size_t i = 0; i < wcoeffs.extent(0); ++i)
  {
    for (std::size_t j = 0; j < i; ++j)
    {
      double a = 0;
      for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
        a += wcoeffs(i, k) * wcoeffs(j, k);
      for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
        wcoeffs(i, k) -= a * wcoeffs(j, k);
    }

    double norm = 0.0;
    for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
      norm += wcoeffs(i, k) * wcoeffs(i, k);

    for (std::size_t k = 0; k < wcoeffs.extent(1); ++k)
      wcoeffs(i, k) /= std::sqrt(norm);
  }
}
//-----------------------------------------------------------------------------
constexpr int compute_value_size(maps::type map_type, int dim)
{
  switch (map_type)
  {
  case maps::type::identity:
    return 1;
  case maps::type::covariantPiola:
    return dim;
  case maps::type::contravariantPiola:
    return dim;
  case maps::type::doubleCovariantPiola:
    return dim * dim;
  case maps::type::doubleContravariantPiola:
    return dim * dim;
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}
//-----------------------------------------------------------------------------
constexpr int num_transformations(cell::type cell_type)
{
  switch (cell_type)
  {
  case cell::type::point:
    return 0;
  case cell::type::interval:
    return 0;
  case cell::type::triangle:
    return 3;
  case cell::type::quadrilateral:
    return 4;
  case cell::type::tetrahedron:
    return 14;
  case cell::type::hexahedron:
    return 24;
  case cell::type::prism:
    return 19;
  case cell::type::pyramid:
    return 18;
  default:
    throw std::runtime_error("Cell type not yet supported");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
compute_dual_matrix(cell::type cell_type, cmdspan2_t B,
                    const std::array<std::vector<impl::cmdspan2_t>, 4>& x,
                    const std::array<std::vector<impl::cmdspan4_t>, 4>& M,
                    int degree, int nderivs)
{
  std::size_t num_dofs(0), vs(0);
  for (auto& Md : M)
  {
    for (auto& Me : Md)
    {
      num_dofs += Me.extent(0);
      if (vs == 0)
        vs = Me.extent(1);
      else if (vs != Me.extent(1))
        throw std::runtime_error("Inconsistent value size");
    }
  }

  std::size_t pdim = polyset::dim(cell_type, degree);
  mdarray3_t D(vs, pdim, num_dofs);
  std::fill(D.data(), D.data() + D.size(), 0);
  std::vector<double> Pb;

  // Loop over different dimensions
  std::size_t dof_index = 0;
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    // Loop over entities of dimension d
    for (std::size_t e = 0; e < x[d].size(); ++e)
    {
      // Evaluate polynomial basis at x[d]
      cmdspan2_t x_e = x[d][e];
      cmdspan3_t P;
      if (x_e.extent(0) > 0)
      {
        std::array<std::size_t, 3> shape;
        std::tie(Pb, shape)
            = polyset::tabulate(cell_type, degree, nderivs, x_e);
        P = cmdspan3_t(Pb.data(), shape);
      }

      // Me: [dof, vs, point, deriv]
      cmdspan4_t Me = M[d][e];

      // Compute dual matrix contribution
      if (Me.extent(3) > 1)
      {
        for (std::size_t l = 0; l < Me.extent(3); ++l)       // Derivative
          for (std::size_t m = 0; m < P.extent(1); ++m)      // Polynomial term
            for (std::size_t i = 0; i < Me.extent(0); ++i)   // Dof index
              for (std::size_t j = 0; j < Me.extent(1); ++j) // Value index
                for (std::size_t k = 0; k < Me.extent(2); ++k) // Point
                  D(j, m, dof_index + i) += Me(i, j, k, l) * P(l, m, k);
      }
      else
      {
        // Flatten and use matrix-matrix multiplication, possibly using
        // BLAS for larger cases. We can do this straightforwardly when
        // Me.extent(3) == 1 since we are contracting over one index
        // only.

        std::vector<double> Pt_b(P.extent(2) * P.extent(1));
        mdspan2_t Pt(Pt_b.data(), P.extent(2), P.extent(1));
        for (std::size_t i = 0; i < Pt.extent(0); ++i)
          for (std::size_t j = 0; j < Pt.extent(1); ++j)
            Pt(i, j) = P(0, j, i);

        std::vector<double> De_b(Me.extent(0) * Me.extent(1) * Pt.extent(1));
        mdspan2_t De(De_b.data(), Me.extent(0) * Me.extent(1), Pt.extent(1));
        math::dot(cmdspan2_t(Me.data_handle(), Me.extent(0) * Me.extent(1),
                             Me.extent(2)),
                  Pt, De);

        // Expand and copy
        for (std::size_t i = 0; i < Me.extent(0); ++i)
          for (std::size_t j = 0; j < Me.extent(1); ++j)
            for (std::size_t k = 0; k < P.extent(1); ++k)
              D(j, k, dof_index + i) += De(i * Me.extent(1) + j, k);
      }

      dof_index += M[d][e].extent(0);
    }
  }

  // Flatten D
  cmdspan2_t Df(D.data(), D.extent(0) * D.extent(1), D.extent(2));

  std::array shape = {B.extent(0), Df.extent(1)};
  std::vector<double> C(shape[0] * shape[1]);
  math::dot(B, Df, mdspan2_t(C.data(), shape));
  return {std::move(C), shape};
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree,
                                           element::lagrange_variant lvariant,
                                           bool discontinuous)
{
  return create_element(family, cell, degree, lvariant,
                        element::dpc_variant::unset, discontinuous);
}

basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree,
                                           element::lagrange_variant lvariant,
                                           element::dpc_variant dvariant,
                                           bool discontinuous)
{
  if (family == element::family::custom)
  {
    throw std::runtime_error("Cannot create a custom element directly. Try "
                             "using `create_custom_element` instead");
  }

  if (degree < 0)
  {
    throw std::runtime_error("Cannot create an element with a negative degree");
  }

  // Checklist of variant compatibility (lagrange, DPC) for each
  static const std::map<element::family, std::array<bool, 2>> has_variant
      = {{element::family::P, {true, false}},
         {element::family::RT, {true, false}},
         {element::family::N1E, {true, false}},
         {element::family::serendipity, {true, true}},
         {element::family::DPC, {false, true}},
         {element::family::Regge, {false, false}},
         {element::family::HHJ, {false, false}},
         {element::family::CR, {false, false}},
         {element::family::bubble, {false, false}},
         {element::family::Hermite, {false, false}}};
  if (auto it = has_variant.find(family); it != has_variant.end())
  {
    if (it->second[0] == false and lvariant != element::lagrange_variant::unset)
    {
      throw std::runtime_error(
          "Cannot pass a Lagrange variant to this element.");
    }
    if (it->second[1] == false and dvariant != element::dpc_variant::unset)
      throw std::runtime_error("Cannot pass a DPC variant to this element.");
  }

  switch (family)
  {
  // P family
  case element::family::P:
    return element::create_lagrange(cell, degree, lvariant, discontinuous);
  case element::family::RT:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_rtc(cell, degree, lvariant, discontinuous);
    case cell::type::hexahedron:
      return element::create_rtc(cell, degree, lvariant, discontinuous);
    default:
      return element::create_rt(cell, degree, lvariant, discontinuous);
    }
  }
  case element::family::N1E:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_nce(cell, degree, lvariant, discontinuous);
    case cell::type::hexahedron:
      return element::create_nce(cell, degree, lvariant, discontinuous);
    default:
      return element::create_nedelec(cell, degree, lvariant, discontinuous);
    }
  }
  // S family
  case element::family::serendipity:
    return element::create_serendipity(cell, degree, lvariant, dvariant,
                                       discontinuous);
  case element::family::BDM:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_serendipity_div(cell, degree, lvariant, dvariant,
                                             discontinuous);
    case cell::type::hexahedron:
      return element::create_serendipity_div(cell, degree, lvariant, dvariant,
                                             discontinuous);
    default:
      return element::create_bdm(cell, degree, lvariant, discontinuous);
    }
  case element::family::N2E:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_serendipity_curl(cell, degree, lvariant, dvariant,
                                              discontinuous);
    case cell::type::hexahedron:
      return element::create_serendipity_curl(cell, degree, lvariant, dvariant,
                                              discontinuous);
    default:
      return element::create_nedelec2(cell, degree, lvariant, discontinuous);
    }
  case element::family::DPC:
    return element::create_dpc(cell, degree, dvariant, discontinuous);

  // Matrix elements
  case element::family::Regge:
    return element::create_regge(cell, degree, discontinuous);
  case element::family::HHJ:
    return element::create_hhj(cell, degree, discontinuous);

  // Other elements
  case element::family::CR:
    return element::create_cr(cell, degree, discontinuous);
  case element::family::bubble:
    return element::create_bubble(cell, degree, discontinuous);
  case element::family::Hermite:
    return element::create_hermite(cell, degree, discontinuous);
  default:
    throw std::runtime_error("Element family not found.");
  }
}
//-----------------------------------------------------------------------------
std::tuple<std::array<std::vector<std::vector<double>>, 4>,
           std::array<std::vector<std::array<std::size_t, 2>>, 4>,
           std::array<std::vector<std::vector<double>>, 4>,
           std::array<std::vector<std::array<std::size_t, 4>>, 4>>
element::make_discontinuous(const std::array<std::vector<cmdspan2_t>, 4>& x,
                            const std::array<std::vector<cmdspan4_t>, 4>& M,
                            std::size_t tdim, std::size_t value_size)
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

  std::array<std::vector<std::vector<double>>, 4> x_data;
  std::array<std::vector<std::array<std::size_t, 2>>, 4> xshapes;
  std::array<std::vector<std::vector<double>>, 4> M_data;
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
  std::vector<double> xb(xshape[0] * xshape[1]);
  stdex::mdspan<double, stdex::dextents<std::size_t, 2>> new_x(xb.data(),
                                                               xshape);

  std::array<std::size_t, 4> Mshape = {Mshape0, value_size, npoints, nderivs};
  std::vector<double> Mb(Mshape[0] * Mshape[1] * Mshape[2] * Mshape[3]);
  stdex::mdspan<double, stdex::dextents<std::size_t, 4>> new_M(Mb.data(),
                                                               Mshape);

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
basix::FiniteElement basix::create_custom_element(
    cell::type cell_type, const std::vector<std::size_t>& value_shape,
    const impl::cmdspan2_t& wcoeffs,
    const std::array<std::vector<impl::cmdspan2_t>, 4>& x,
    const std::array<std::vector<impl::cmdspan4_t>, 4>& M,
    int interpolation_nderivs, maps::type map_type,
    sobolev::space sobolev_space, bool discontinuous,
    int highest_complete_degree, int highest_degree)
{
  // Check that inputs are valid
  const std::size_t psize = polyset::dim(cell_type, highest_degree);
  const std::size_t value_size = std::reduce(
      value_shape.begin(), value_shape.end(), 1, std::multiplies{});

  const std::size_t deriv_count
      = polyset::nderivs(cell_type, interpolation_nderivs);

  const std::size_t tdim = cell::topological_dimension(cell_type);

  std::size_t ndofs = 0;
  for (std::size_t i = 0; i <= 3; ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      ndofs += M[i][j].extent(0);

  // Check that wcoeffs have the correct shape
  if (wcoeffs.extent(1) != psize * value_size)
    throw std::runtime_error("wcoeffs has the wrong number of columns");
  if (wcoeffs.extent(0) != ndofs)
    throw std::runtime_error("wcoeffs has the wrong number of rows");

  // Check that x has the right shape
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    if (x[i].size()
        != (i > tdim ? 0
                     : static_cast<std::size_t>(
                         cell::num_sub_entities(cell_type, i))))
    {
      throw std::runtime_error("x has the wrong number of entities");
    }

    for (const auto& xj : x[i])
    {
      if (xj.extent(1) != tdim)
        throw std::runtime_error("x has a point with the wrong tdim");
    }
  }

  // Check that M has the right shape
  for (std::size_t i = 0; i < M.size(); ++i)
  {
    if (M[i].size()
        != (i > tdim ? 0
                     : static_cast<std::size_t>(
                         cell::num_sub_entities(cell_type, i))))
      throw std::runtime_error("M has the wrong number of entities");
    for (std::size_t j = 0; j < M[i].size(); ++j)
    {
      if (M[i][j].extent(2) != x[i][j].extent(0))
      {
        throw std::runtime_error(
            "M has the wrong shape (dimension 2 is wrong)");
      }
      if (M[i][j].extent(1) != value_size)
      {
        throw std::runtime_error(
            "M has the wrong shape (dimension 1 is wrong)");
      }
      if (M[i][j].extent(3) != deriv_count)
      {
        throw std::runtime_error(
            "M has the wrong shape (dimension 3 is wrong)");
      }
    }
  }

  auto [dualmatrix, dualshape] = compute_dual_matrix(
      cell_type, wcoeffs, x, M, highest_degree, interpolation_nderivs);
  if (math::is_singular(cmdspan2_t(dualmatrix.data(), dualshape)))
  {
    throw std::runtime_error(
        "Dual matrix is singular, there is an error in your inputs");
  }

  return basix::FiniteElement(
      element::family::custom, cell_type, highest_degree, value_shape, wcoeffs,
      x, M, interpolation_nderivs, map_type, sobolev_space, discontinuous,
      highest_complete_degree, highest_degree, element::lagrange_variant::unset,
      element::dpc_variant::unset);
}

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(
    element::family family, cell::type cell_type, int degree,
    const std::vector<std::size_t>& value_shape, const cmdspan2_t& wcoeffs,
    const std::array<std::vector<cmdspan2_t>, 4>& x,
    const std::array<std::vector<cmdspan4_t>, 4>& M, int interpolation_nderivs,
    maps::type map_type, sobolev::space sobolev_space, bool discontinuous,
    int highest_complete_degree, int highest_degree,
    element::lagrange_variant lvariant, element::dpc_variant dvariant,
    std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
        tensor_factors)
    : _cell_type(cell_type), _cell_tdim(cell::topological_dimension(cell_type)),
      _cell_subentity_types(cell::subentity_types(cell_type)), _family(family),
      _lagrange_variant(lvariant), _dpc_variant(dvariant), _degree(degree),
      _interpolation_nderivs(interpolation_nderivs),
      _highest_degree(highest_degree),
      _highest_complete_degree(highest_complete_degree),
      _value_shape(value_shape), _map_type(map_type),
      _sobolev_space(sobolev_space), _discontinuous(discontinuous),
      _tensor_factors(tensor_factors)
{
  // Check that discontinuous elements only have DOFs on interior
  if (discontinuous)
  {
    for (std::size_t i = 0; i < _cell_tdim; ++i)
    {
      for (auto& xi : x[i])
      {
        if (xi.extent(0) > 0)
        {
          throw std::runtime_error(
              "Discontinuous element can only have interior DOFs.");
        }
      }
    }
  }

  // Copy x
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    for (auto& xi : x[i])
    {
      _x[i].emplace_back(
          std::vector(xi.data_handle(), xi.data_handle() + xi.size()),
          std::array{xi.extent(0), xi.extent(1)});
    }
  }

  std::vector<double> wcoeffs_ortho_b(wcoeffs.extent(0) * wcoeffs.extent(1));
  mdspan2_t wcoeffs_ortho(wcoeffs_ortho_b.data(), wcoeffs.extent(0),
                          wcoeffs.extent(1));
  std::copy(wcoeffs.data_handle(), wcoeffs.data_handle() + wcoeffs.size(),
            wcoeffs_ortho_b.begin());
  if (family != element::family::P)
    orthogonalise(wcoeffs_ortho);
  _dual_matrix = compute_dual_matrix(cell_type, wcoeffs_ortho, x, M,
                                     highest_degree, interpolation_nderivs);

  _wcoeffs
      = {wcoeffs_ortho_b, {wcoeffs_ortho.extent(0), wcoeffs_ortho.extent(1)}};

  // Copy  M
  for (std::size_t i = 0; i < M.size(); ++i)
  {
    for (auto Mi : M[i])
    {
      _M[i].emplace_back(
          std::vector(Mi.data_handle(), Mi.data_handle() + Mi.size()),
          std::array{Mi.extent(0), Mi.extent(1), Mi.extent(2), Mi.extent(3)});
    }
  }

  // Compute C = (BD^T)^{-1} B
  _coeffs.first
      = math::solve(cmdspan2_t(_dual_matrix.first.data(), _dual_matrix.second),
                    wcoeffs_ortho);
  _coeffs.second = {_dual_matrix.second[1], wcoeffs_ortho.extent(1)};

  std::size_t num_points = 0;
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      num_points += x_e.extent(0);

  _points.first.reserve(num_points * _cell_tdim);
  _points.second = {num_points, _cell_tdim};
  mdspan2_t pview(_points.first.data(), _points.second);
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      for (std::size_t p = 0; p < x_e.extent(0); ++p)
        for (std::size_t k = 0; k < x_e.extent(1); ++k)
          _points.first.push_back(x_e(p, k));

  // Copy into _matM
  const std::size_t value_size = std::accumulate(
      value_shape.begin(), value_shape.end(), 1, std::multiplies{});

  // Count number of dofs and point
  std::size_t num_dofs(0), num_points1(0);
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    for (auto Me : M[d])
    {
      num_dofs += Me.extent(0);
      num_points1 += Me.extent(2);
    }
  }

  // Check that number of dofs is equal to number of coefficients
  if (num_dofs != _coeffs.second[0])
  {
    throw std::runtime_error(
        "Number of entity dofs does not match total number of dofs");
  }

  _entity_transformations = doftransforms::compute_entity_transformations(
      cell_type, x, M, cmdspan2_t(_coeffs.first.data(), _coeffs.second),
      highest_degree, value_size, map_type);

  const std::size_t nderivs
      = polyset::nderivs(cell_type, interpolation_nderivs);

  _matM = {std::vector<double>(num_dofs * value_size * num_points1 * nderivs),
           {num_dofs, value_size * num_points1 * nderivs}};
  mdspan4_t Mview(_matM.first.data(), num_dofs, value_size, num_points1,
                  nderivs);

  // Loop over each topological dimensions
  std::size_t dof_offset(0), point_offset(0);
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    // Loop of entities of dimension d
    for (auto& Me : M[d])
    {
      for (std::size_t k0 = 0; k0 < Me.extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < Mview.extent(1); ++k1)
          for (std::size_t k2 = 0; k2 < Me.extent(2); ++k2)
            for (std::size_t k3 = 0; k3 < Mview.extent(3); ++k3)
              Mview(k0 + dof_offset, k1, k2 + point_offset, k3)
                  = Me(k0, k1, k2, k3);

      dof_offset += Me.extent(0);
      point_offset += Me.extent(2);
    }
  }

  // Compute number of dofs for each cell entity (computed from
  // interpolation data)
  int dof = 0;
  for (std::size_t d = 0; d < _cell_tdim + 1; ++d)
  {
    auto& edofs_d = _edofs.emplace_back(cell::num_sub_entities(_cell_type, d));
    for (std::size_t e = 0; e < M[d].size(); ++e)
      for (std::size_t i = 0; i < M[d][e].extent(0); ++i)
        edofs_d[e].push_back(dof++);
  }

  const std::vector<std::vector<std::vector<std::vector<int>>>> connectivity
      = cell::sub_entity_connectivity(cell_type);
  for (std::size_t d = 0; d < _cell_tdim + 1; ++d)
  {
    auto& edofs_d
        = _e_closure_dofs.emplace_back(cell::num_sub_entities(_cell_type, d));
    for (std::size_t e = 0; e < _e_closure_dofs[d].size(); ++e)
    {
      auto& closure_dofs = edofs_d[e];
      for (std::size_t dim = 0; dim <= d; ++dim)
      {
        for (int c : connectivity[d][e][dim])
        {
          closure_dofs.insert(closure_dofs.end(), _edofs[dim][c].begin(),
                              _edofs[dim][c].end());
        }
      }

      std::sort(_e_closure_dofs[d][e].begin(), _e_closure_dofs[d][e].end());
    }
  }

  // Check if base transformations are all permutations
  _dof_transformations_are_permutations = true;
  _dof_transformations_are_identity = true;
  for (const auto& [ctype, trans_data] : _entity_transformations)
  {
    cmdspan3_t trans(trans_data.first.data(), trans_data.second);

    for (std::size_t i = 0;
         _dof_transformations_are_permutations and i < trans.extent(0); ++i)
    {
      for (std::size_t row = 0; row < trans.extent(1); ++row)
      {
        double rmin(0), rmax(0), rtot(0);
        for (std::size_t k = 0; k < trans.extent(2); ++k)
        {
          double r = trans(i, row, k);
          rmin = std::min(r, rmin);
          rmax = std::max(r, rmax);
          rtot += r;
        }

        if ((trans.extent(2) != 1 and std::abs(rmin) > 1.0e-8)
            or std::abs(rmax - 1.0) > 1.0e-8 or std::abs(rtot - 1.0) > 1.0e-8)
        {
          _dof_transformations_are_permutations = false;
          _dof_transformations_are_identity = false;
          break;
        }

        if (std::abs(trans(i, row, row) - 1) > 1.0e-8)
          _dof_transformations_are_identity = false;
      }
    }
    if (!_dof_transformations_are_permutations)
      break;
  }

  if (!_dof_transformations_are_identity)
  {
    // If transformations are permutations, then create the permutations
    if (_dof_transformations_are_permutations)
    {
      for (const auto& [ctype, trans_data] : _entity_transformations)
      {
        cmdspan3_t trans(trans_data.first.data(), trans_data.second);

        for (std::size_t i = 0; i < trans.extent(0); ++i)
        {
          std::vector<std::size_t> perm(trans.extent(1));
          std::vector<std::size_t> rev_perm(trans.extent(1));
          for (std::size_t row = 0; row < trans.extent(1); ++row)
          {
            for (std::size_t col = 0; col < trans.extent(1); ++col)
            {
              if (trans(i, row, col) > 0.5)
              {
                perm[row] = col;
                rev_perm[col] = row;
                break;
              }
            }
          }

          // Factorise the permutations
          precompute::prepare_permutation(perm);
          precompute::prepare_permutation(rev_perm);

          // Store the permutations
          auto& eperm = _eperm.try_emplace(ctype).first->second;
          auto& eperm_rev = _eperm_rev.try_emplace(ctype).first->second;
          eperm.push_back(perm);
          eperm_rev.push_back(rev_perm);

          // Generate the entity transformations from the permutations
          std::pair<std::vector<double>, std::array<std::size_t, 2>> identity
              = {std::vector<double>(perm.size() * perm.size()),
                 {perm.size(), perm.size()}};
          std::fill(identity.first.begin(), identity.first.end(), 0.);
          for (std::size_t i = 0; i < perm.size(); ++i)
            identity.first[i * perm.size() + i] = 1;

          auto& etrans = _etrans.try_emplace(ctype).first->second;
          auto& etransT = _etransT.try_emplace(ctype).first->second;
          auto& etrans_invT = _etrans_invT.try_emplace(ctype).first->second;
          auto& etrans_inv = _etrans_inv.try_emplace(ctype).first->second;
          etrans.push_back({perm, identity});
          etrans_invT.push_back({perm, identity});
          etransT.push_back({rev_perm, identity});
          etrans_inv.push_back({rev_perm, identity});
        }
      }
    }
    else
    {

      // Precompute the DOF transformations
      for (const auto& [ctype, trans_data] : _entity_transformations)
      {
        cmdspan3_t trans(trans_data.first.data(), trans_data.second);

        // Buffers for matrices
        std::vector<double> M_b, Minv_b, matint;

        auto& etrans = _etrans.try_emplace(ctype).first->second;
        auto& etransT = _etransT.try_emplace(ctype).first->second;
        auto& etrans_invT = _etrans_invT.try_emplace(ctype).first->second;
        auto& etrans_inv = _etrans_inv.try_emplace(ctype).first->second;
        for (std::size_t i = 0; i < trans.extent(0); ++i)
        {
          if (trans.extent(1) == 0)
          {
            etrans.push_back({});
            etransT.push_back({});
            etrans_invT.push_back({});
            etrans_inv.push_back({});
          }
          else
          {
            const std::size_t dim = trans.extent(1);
            assert(dim == trans.extent(2));

            {
              std::pair<std::vector<double>, std::array<std::size_t, 2>> mat
                  = {std::vector<double>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  mat.first[k0 * dim + k1] = trans(i, k0, k1);
              std::vector<std::size_t> mat_p = precompute::prepare_matrix(mat);
              etrans.push_back({mat_p, mat});
            }

            {
              std::pair<std::vector<double>, std::array<std::size_t, 2>> matT
                  = {std::vector<double>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  matT.first[k0 * dim + k1] = trans(i, k1, k0);
              std::vector<std::size_t> matT_p
                  = precompute::prepare_matrix(matT);
              etransT.push_back({matT_p, matT});
            }

            M_b.resize(dim * dim);
            mdspan2_t M(M_b.data(), dim, dim);
            for (std::size_t k0 = 0; k0 < dim; ++k0)
              for (std::size_t k1 = 0; k1 < dim; ++k1)
                M(k0, k1) = trans(i, k0, k1);

            // Rotation of a face: this is in the only base transformation
            // such that M^{-1} != M.
            // For a quadrilateral face, M^4 = Id, so M^{-1} = M^3.
            // For a triangular face, M^3 = Id, so M^{-1} = M^2.
            Minv_b.resize(dim * dim);
            mdspan2_t Minv(Minv_b.data(), dim, dim);
            if (ctype == cell::type::quadrilateral and i == 0)
            {
              matint.resize(dim * dim);
              mdspan2_t mat_int(matint.data(), dim, dim);
              math::dot(M, M, mat_int);

              math::dot(mat_int, M, Minv);
            }
            else if (ctype == cell::type::triangle and i == 0)
            {
              math::dot(M, M, Minv);
            }
            else
            {
              Minv_b.assign(M_b.begin(), M_b.end());
            }

            {
              std::pair<std::vector<double>, std::array<std::size_t, 2>> mat_inv
                  = {std::vector<double>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  mat_inv.first[k0 * dim + k1] = Minv(k0, k1);
              std::vector<std::size_t> mat_inv_p
                  = precompute::prepare_matrix(mat_inv);
              etrans_inv.push_back({mat_inv_p, mat_inv});
            }

            {
              std::pair<std::vector<double>, std::array<std::size_t, 2>>
                  mat_invT = {std::vector<double>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  mat_invT.first[k0 * dim + k1] = Minv(k1, k0);
              std::vector<std::size_t> mat_invT_p
                  = precompute::prepare_matrix(mat_invT);
              etrans_invT.push_back({mat_invT_p, mat_invT});
            }
          }
        }
      }
    }
  }

  // Check if interpolation matrix is the identity
  cmdspan2_t matM(_matM.first.data(), _matM.second);
  _interpolation_is_identity = matM.extent(0) == matM.extent(1);
  for (std::size_t row = 0; _interpolation_is_identity && row < matM.extent(0);
       ++row)
  {
    for (std::size_t col = 0; col < matM.extent(1); ++col)
    {
      double v = col == row ? 1.0 : 0.0;
      if (std::abs(matM(row, col) - v) > 1.0e-12)
      {
        _interpolation_is_identity = false;
        break;
      }
    }
  }
}
//-----------------------------------------------------------------------------
bool FiniteElement::operator==(const FiniteElement& e) const
{
  if (this == &e)
    return true;
  else if (family() == element::family::custom
           and e.family() == element::family::custom)
  {
    bool coeff_equal = false;
    if (_coeffs.first.size() == e.coefficient_matrix().first.size()
        and _coeffs.second == e.coefficient_matrix().second
        and std::equal(_coeffs.first.begin(), _coeffs.first.end(),
                       e.coefficient_matrix().first.begin(),
                       [](auto x, auto y)
                       { return std::abs(x - y) < 1.0e-10; }))
    {
      coeff_equal = true;
    }

    return cell_type() == e.cell_type() and discontinuous() == e.discontinuous()
           and map_type() == e.map_type()
           and sobolev_space() == e.sobolev_space()
           and value_shape() == e.value_shape()
           and highest_degree() == e.highest_degree()
           and highest_complete_degree() == e.highest_complete_degree()
           and coeff_equal and entity_dofs() == e.entity_dofs();
  }
  else
  {
    return cell_type() == e.cell_type() and family() == e.family()
           and degree() == e.degree() and discontinuous() == e.discontinuous()
           and lagrange_variant() == e.lagrange_variant()
           and dpc_variant() == e.dpc_variant() and map_type() == e.map_type()
           and sobolev_space() == e.sobolev_space();
  }
}
//-----------------------------------------------------------------------------
std::array<std::size_t, 4>
FiniteElement::tabulate_shape(std::size_t nd, std::size_t num_points) const
{
  std::size_t ndsize = 1;
  for (std::size_t i = 1; i <= nd; ++i)
    ndsize *= (_cell_tdim + i);
  for (std::size_t i = 1; i <= nd; ++i)
    ndsize /= i;
  std::size_t vs = std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                                   std::multiplies{});
  std::size_t ndofs = _coeffs.second[0];
  return {ndsize, num_points, ndofs, vs};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 4>>
FiniteElement::tabulate(int nd, impl::cmdspan2_t x) const
{
  std::array<std::size_t, 4> shape = tabulate_shape(nd, x.extent(0));
  std::vector<double> data(shape[0] * shape[1] * shape[2] * shape[3]);
  tabulate(nd, x, mdspan4_t(data.data(), shape));
  return {std::move(data), shape};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 4>>
FiniteElement::tabulate(int nd, const std::span<const double>& x,
                        std::array<std::size_t, 2> shape) const
{
  std::array<std::size_t, 4> phishape = tabulate_shape(nd, shape[0]);
  std::vector<double> datab(phishape[0] * phishape[1] * phishape[2]
                            * phishape[3]);
  tabulate(nd, cmdspan2_t(x.data(), shape[0], shape[1]),
           mdspan4_t(datab.data(), phishape));
  return {std::move(datab), phishape};
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate(int nd, impl::cmdspan2_t x,
                             impl::mdspan4_t basis_data) const
{
  if (x.extent(1) != _cell_tdim)
  {
    throw std::runtime_error("Point dim (" + std::to_string(x.extent(1))
                             + ") does not match element dim ("
                             + std::to_string(_cell_tdim) + ").");
  }

  const std::size_t psize = polyset::dim(_cell_type, _highest_degree);
  const std::array<std::size_t, 3> bsize
      = {(std::size_t)polyset::nderivs(_cell_type, nd), psize, x.extent(0)};
  std::vector<double> basis_b(bsize[0] * bsize[1] * bsize[2]);
  mdspan3_t basis(basis_b.data(), bsize);
  polyset::tabulate(basis, _cell_type, _highest_degree, nd, x);
  const int vs = std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                                 std::multiplies{});

  std::vector<double> C_b(_coeffs.second[0] * psize);
  mdspan2_t C(C_b.data(), _coeffs.second[0], psize);

  cmdspan2_t coeffs_view(_coeffs.first.data(), _coeffs.second);
  std::vector<double> result_b(C.extent(0) * bsize[2]);
  mdspan2_t result(result_b.data(), C.extent(0), bsize[2]);
  for (std::size_t p = 0; p < basis.extent(0); ++p)
  {
    cmdspan2_t B(basis_b.data() + p * bsize[1] * bsize[2], bsize[1], bsize[2]);
    for (int j = 0; j < vs; ++j)
    {
      for (std::size_t k0 = 0; k0 < coeffs_view.extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < psize; ++k1)
          C(k0, k1) = coeffs_view(k0, k1 + psize * j);

      math::dot(C, cmdspan2_t(B.data_handle(), B.extent(0), B.extent(1)),
                result);

      for (std::size_t k0 = 0; k0 < basis_data.extent(1); ++k0)
        for (std::size_t k1 = 0; k1 < basis_data.extent(2); ++k1)
          basis_data(p, k0, k1, j) = result(k1, k0);
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate(int nd, const std::span<const double>& x,
                             std::array<std::size_t, 2> xshape,
                             const std::span<double>& basis) const
{
  std::array<std::size_t, 4> shape = tabulate_shape(nd, xshape[0]);
  assert(x.size() == xshape[0] * xshape[1]);
  assert(basis.size() == shape[0] * shape[1] * shape[2] * shape[3]);
  tabulate(nd, cmdspan2_t(x.data(), xshape), mdspan4_t(basis.data(), shape));
}
//-----------------------------------------------------------------------------
cell::type FiniteElement::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
int FiniteElement::degree() const { return _degree; }
//-----------------------------------------------------------------------------
int FiniteElement::highest_degree() const { return _highest_degree; }
//-----------------------------------------------------------------------------
int FiniteElement::highest_complete_degree() const
{
  return _highest_complete_degree;
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& FiniteElement::value_shape() const
{
  return _value_shape;
}
//-----------------------------------------------------------------------------
int FiniteElement::dim() const { return _coeffs.second[0]; }
//-----------------------------------------------------------------------------
element::family FiniteElement::family() const { return _family; }
//-----------------------------------------------------------------------------
maps::type FiniteElement::map_type() const { return _map_type; }
//-----------------------------------------------------------------------------
sobolev::space FiniteElement::sobolev_space() const { return _sobolev_space; }
//-----------------------------------------------------------------------------
bool FiniteElement::discontinuous() const { return _discontinuous; }
//-----------------------------------------------------------------------------
bool FiniteElement::dof_transformations_are_permutations() const
{
  return _dof_transformations_are_permutations;
}
//-----------------------------------------------------------------------------
bool FiniteElement::dof_transformations_are_identity() const
{
  return _dof_transformations_are_identity;
}
//-----------------------------------------------------------------------------
const std::pair<std::vector<double>, std::array<std::size_t, 2>>&
FiniteElement::interpolation_matrix() const
{
  return _matM;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
FiniteElement::entity_dofs() const
{
  return _edofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
FiniteElement::entity_closure_dofs() const
{
  return _e_closure_dofs;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 3>>
FiniteElement::base_transformations() const
{
  const std::size_t nt = num_transformations(this->cell_type());
  const std::size_t ndofs = this->dim();

  std::array<std::size_t, 3> shape = {nt, ndofs, ndofs};
  std::vector<double> bt_b(shape[0] * shape[1] * shape[2], 0);
  mdspan3_t bt(bt_b.data(), shape);
  for (std::size_t i = 0; i < nt; ++i)
    for (std::size_t j = 0; j < ndofs; ++j)
      bt(i, j, j) = 1.0;

  std::size_t dofstart = 0;
  if (_cell_tdim > 0)
  {
    for (auto& edofs0 : _edofs[0])
      dofstart += edofs0.size();
  }

  int transform_n = 0;
  if (_cell_tdim > 1)
  {
    // Base transformations for edges
    {
      auto& tmp_data = _entity_transformations.at(cell::type::interval);
      cmdspan3_t tmp(tmp_data.first.data(), tmp_data.second);
      for (auto& e : _edofs[1])
      {
        std::size_t ndofs = e.size();
        for (std::size_t i = 0; i < ndofs; ++i)
          for (std::size_t j = 0; j < ndofs; ++j)
            bt(transform_n, i + dofstart, j + dofstart) = tmp(0, i, j);

        ++transform_n;
        dofstart += ndofs;
      }
    }

    if (_cell_tdim > 2)
    {
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        if (std::size_t ndofs = _edofs[2][f].size(); ndofs > 0)
        {
          auto& tmp_data
              = _entity_transformations.at(_cell_subentity_types[2][f]);
          cmdspan3_t tmp(tmp_data.first.data(), tmp_data.second);

          for (std::size_t i = 0; i < ndofs; ++i)
            for (std::size_t j = 0; j < ndofs; ++j)
              bt(transform_n, i + dofstart, j + dofstart) = tmp(0, i, j);
          ++transform_n;

          for (std::size_t i = 0; i < ndofs; ++i)
            for (std::size_t j = 0; j < ndofs; ++j)
              bt(transform_n, i + dofstart, j + dofstart) = tmp(1, i, j);
          ++transform_n;

          dofstart += ndofs;
        }
      }
    }
  }

  return {std::move(bt_b), shape};
}
//-----------------------------------------------------------------------------
const std::pair<std::vector<double>, std::array<std::size_t, 2>>&
FiniteElement::points() const
{
  return _points;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 3>>
FiniteElement::push_forward(impl::cmdspan3_t U, impl::cmdspan3_t J,
                            std::span<const double> detJ,
                            impl::cmdspan3_t K) const
{
  const std::size_t physical_value_size
      = compute_value_size(_map_type, J.extent(1));

  std::array<std::size_t, 3> shape
      = {U.extent(0), U.extent(1), physical_value_size};
  std::vector<double> ub(shape[0] * shape[1] * shape[2]);
  mdspan3_t u(ub.data(), shape);

  using u_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using U_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using J_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using K_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;

  auto map = this->map_fn<u_t, U_t, J_t, K_t>();
  for (std::size_t i = 0; i < u.extent(0); ++i)
  {
    u_t _u(u.data_handle() + i * u.extent(1) * u.extent(2), u.extent(1),
           u.extent(2));
    U_t _U(U.data_handle() + i * U.extent(1) * U.extent(2), U.extent(1),
           U.extent(2));
    J_t _J(J.data_handle() + i * J.extent(1) * J.extent(2), J.extent(1),
           J.extent(2));
    K_t _K(K.data_handle() + i * K.extent(1) * K.extent(2), K.extent(1),
           K.extent(2));
    map(_u, _U, _J, detJ[i], _K);
  }

  return {std::move(ub), shape};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 3>>
FiniteElement::pull_back(impl::cmdspan3_t u, impl::cmdspan3_t J,
                         std::span<const double> detJ, impl::cmdspan3_t K) const
{
  const std::size_t reference_value_size = std::accumulate(
      _value_shape.begin(), _value_shape.end(), 1, std::multiplies{});

  std::array<std::size_t, 3> shape
      = {u.extent(0), u.extent(1), reference_value_size};
  std::vector<double> Ub(shape[0] * shape[1] * shape[2]);
  mdspan3_t U(Ub.data(), shape);

  using u_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using U_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using J_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using K_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  auto map = this->map_fn<U_t, u_t, K_t, J_t>();
  for (std::size_t i = 0; i < u.extent(0); ++i)
  {
    u_t _u(u.data_handle() + i * u.extent(1) * u.extent(2), u.extent(1),
           u.extent(2));
    U_t _U(U.data_handle() + i * U.extent(1) * U.extent(2), U.extent(1),
           U.extent(2));
    J_t _J(J.data_handle() + i * J.extent(1) * J.extent(2), J.extent(1),
           J.extent(2));
    K_t _K(K.data_handle() + i * K.extent(1) * K.extent(2), K.extent(1),
           K.extent(2));
    map(_U, _u, _K, 1.0 / detJ[i], _J);
  }

  return {std::move(Ub), shape};
}
//-----------------------------------------------------------------------------
void FiniteElement::permute_dofs(const std::span<std::int32_t>& dofs,
                                 std::uint32_t cell_info) const
{
  if (!_dof_transformations_are_permutations)
  {
    throw std::runtime_error(
        "The DOF transformations for this element are not permutations");
  }

  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if 3D
    // cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _edofs[2].size() : 0;
    int dofstart = 0;
    for (auto& edofs0 : _edofs[0])
      dofstart += edofs0.size();

    // Permute DOFs on edges
    {
      auto& trans = _eperm.at(cell::type::interval)[0];
      for (std::size_t e = 0; e < _edofs[1].size(); ++e)
      {
        // Reverse an edge
        if (cell_info >> (face_start + e) & 1)
          precompute::apply_permutation(trans, dofs, dofstart);
        dofstart += _edofs[1][e].size();
      }
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        auto& trans = _eperm.at(_cell_subentity_types[2][f]);

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_permutation(trans[1], dofs, dofstart);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_permutation(trans[0], dofs, dofstart);

        dofstart += _edofs[2][f].size();
      }
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::unpermute_dofs(const std::span<std::int32_t>& dofs,
                                   std::uint32_t cell_info) const
{
  if (!_dof_transformations_are_permutations)
  {
    throw std::runtime_error(
        "The DOF transformations for this element are not permutations");
  }
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _edofs[2].size() : 0;
    int dofstart = 0;
    for (auto& edofs0 : _edofs[0])
      dofstart += edofs0.size();

    // Permute DOFs on edges
    {
      auto& trans = _eperm_rev.at(cell::type::interval)[0];
      for (std::size_t e = 0; e < _edofs[1].size(); ++e)
      {
        // Reverse an edge
        if (cell_info >> (face_start + e) & 1)
          precompute::apply_permutation(trans, dofs, dofstart);
        dofstart += _edofs[1][e].size();
      }
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        auto& trans = _eperm_rev.at(_cell_subentity_types[2][f]);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_permutation(trans[0], dofs, dofstart);

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_permutation(trans[1], dofs, dofstart);

        dofstart += _edofs[2][f].size();
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::map<cell::type, std::pair<std::vector<double>, std::array<std::size_t, 3>>>
FiniteElement::entity_transformations() const
{
  return _entity_transformations;
}
//-----------------------------------------------------------------------------
const std::pair<std::vector<double>, std::array<std::size_t, 2>>&
FiniteElement::dual_matrix() const
{
  return _dual_matrix;
}
//-----------------------------------------------------------------------------
const std::pair<std::vector<double>, std::array<std::size_t, 2>>&
FiniteElement::wcoeffs() const
{
  return _wcoeffs;
}
//-----------------------------------------------------------------------------
const std::array<
    std::vector<std::pair<std::vector<double>, std::array<std::size_t, 2>>>, 4>&
FiniteElement::x() const
{
  return _x;
}
//-----------------------------------------------------------------------------
const std::array<
    std::vector<std::pair<std::vector<double>, std::array<std::size_t, 4>>>, 4>&
FiniteElement::M() const
{
  return _M;
}
//-----------------------------------------------------------------------------
const std::pair<std::vector<double>, std::array<std::size_t, 2>>&
FiniteElement::coefficient_matrix() const
{
  return _coeffs;
}
//-----------------------------------------------------------------------------
bool FiniteElement::has_tensor_product_factorisation() const
{
  return _tensor_factors.size() > 0;
}
//-----------------------------------------------------------------------------
element::lagrange_variant FiniteElement::lagrange_variant() const
{
  return _lagrange_variant;
}
//-----------------------------------------------------------------------------
element::dpc_variant FiniteElement::dpc_variant() const { return _dpc_variant; }
//-----------------------------------------------------------------------------
bool FiniteElement::interpolation_is_identity() const
{
  return _interpolation_is_identity;
}
//-----------------------------------------------------------------------------
int FiniteElement::interpolation_nderivs() const
{
  return _interpolation_nderivs;
}
//-----------------------------------------------------------------------------
std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
FiniteElement::get_tensor_product_representation() const
{
  if (!has_tensor_product_factorisation())
    throw std::runtime_error("Element has no tensor product representation.");
  return _tensor_factors;
}
//-----------------------------------------------------------------------------
std::string basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
