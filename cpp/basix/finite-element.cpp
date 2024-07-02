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
#include <algorithm>
#include <basix/version.h>
#include <cmath>
#include <concepts>
#include <limits>
#include <numeric>
#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;

namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdarray_t
    = stdex::mdarray<T,
                     MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

namespace
{
//----------------------------------------------------------------------------
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
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>> compute_dual_matrix(
    cell::type cell_type, polyset::type poly_type, mdspan_t<const T, 2> B,
    const std::array<std::vector<impl::mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<impl::mdspan_t<const T, 4>>, 4>& M, int degree,
    int nderivs)
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

  std::size_t pdim = polyset::dim(cell_type, poly_type, degree);
  mdarray_t<T, 3> D(vs, pdim, num_dofs);
  std::fill(D.data(), D.data() + D.size(), 0);
  std::vector<T> Pb;

  // Loop over different dimensions
  std::size_t dof_index = 0;
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    // Loop over entities of dimension d
    for (std::size_t e = 0; e < x[d].size(); ++e)
    {
      // Evaluate polynomial basis at x[d]
      mdspan_t<const T, 2> x_e = x[d][e];
      mdspan_t<const T, 3> P;
      if (x_e.extent(0) > 0)
      {
        std::array<std::size_t, 3> shape;
        std::tie(Pb, shape)
            = polyset::tabulate(cell_type, poly_type, degree, nderivs, x_e);
        P = mdspan_t<const T, 3>(Pb.data(), shape);
      }

      // Me: [dof, vs, point, deriv]
      mdspan_t<const T, 4> Me = M[d][e];

      // Compute dual matrix contribution
      if (Me.extent(3) > 1)
      {
        for (std::size_t l = 0; l < Me.extent(3); ++l)       // Derivative
          for (std::size_t m = 0; m < P.extent(1); ++m)      // Polynomial term
            for (std::size_t i = 0; i < Me.extent(0); ++i)   // Dof index
              for (std::size_t j = 0; j < Me.extent(1); ++j) // Value index
                for (std::size_t k = 0; k < Me.extent(2); ++k) // Point
                  D[j, m, dof_index + i] += Me[i, j, k, l] * P[l, m, k];
      }
      else
      {
        // Flatten and use matrix-matrix multiplication, possibly using
        // BLAS for larger cases. We can do this straightforwardly when
        // Me.extent(3) == 1 since we are contracting over one index
        // only.

        std::vector<T> Pt_b(P.extent(2) * P.extent(1));
        mdspan_t<T, 2> Pt(Pt_b.data(), P.extent(2), P.extent(1));
        for (std::size_t i = 0; i < Pt.extent(0); ++i)
          for (std::size_t j = 0; j < Pt.extent(1); ++j)
            Pt[i, j] = P[0, j, i];

        std::vector<T> De_b(Me.extent(0) * Me.extent(1) * Pt.extent(1));
        mdspan_t<T, 2> De(De_b.data(), Me.extent(0) * Me.extent(1),
                          Pt.extent(1));
        math::dot(mdspan_t<const T, 2>(Me.data_handle(),
                                       Me.extent(0) * Me.extent(1),
                                       Me.extent(2)),
                  Pt, De);

        // Expand and copy
        for (std::size_t i = 0; i < Me.extent(0); ++i)
          for (std::size_t j = 0; j < Me.extent(1); ++j)
            for (std::size_t k = 0; k < P.extent(1); ++k)
              D[j, k, dof_index + i] += De[i * Me.extent(1) + j, k];
      }

      dof_index += M[d][e].extent(0);
    }
  }

  // Flatten D
  mdspan_t<const T, 2> Df(D.data(), D.extent(0) * D.extent(1), D.extent(2));

  std::array shape = {B.extent(0), Df.extent(1)};
  std::vector<T> C(shape[0] * shape[1]);
  math::dot(B, Df, mdspan_t<T, 2>(C.data(), shape));
  return {std::move(C), shape};
}
//-----------------------------------------------------------------------------
void combine_hashes(std::size_t& a, std::size_t b)
{
  a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2);
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>
basix::create_element(element::family family, cell::type cell, int degree,
                      element::lagrange_variant lvariant,
                      element::dpc_variant dvariant, bool discontinuous,
                      std::vector<int> dof_ordering)
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
         {element::family::Hermite, {false, false}},
         {element::family::iso, {true, false}}};
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

  if (!dof_ordering.empty() and family != element::family::P)
  {
    throw std::runtime_error("DOF ordering only supported for Lagrange");
  }

  switch (family)
  {
  // P family
  case element::family::P:
    return element::create_lagrange<T>(cell, degree, lvariant, discontinuous,
                                       dof_ordering);
  case element::family::RT:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_rtc<T>(cell, degree, lvariant, discontinuous);
    case cell::type::hexahedron:
      return element::create_rtc<T>(cell, degree, lvariant, discontinuous);
    default:
      return element::create_rt<T>(cell, degree, lvariant, discontinuous);
    }
  }
  case element::family::N1E:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_nce<T>(cell, degree, lvariant, discontinuous);
    case cell::type::hexahedron:
      return element::create_nce<T>(cell, degree, lvariant, discontinuous);
    default:
      return element::create_nedelec<T>(cell, degree, lvariant, discontinuous);
    }
  }
  // S family
  case element::family::serendipity:
    return element::create_serendipity<T>(cell, degree, lvariant, dvariant,
                                          discontinuous);
  case element::family::BDM:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_serendipity_div<T>(cell, degree, lvariant,
                                                dvariant, discontinuous);
    case cell::type::hexahedron:
      return element::create_serendipity_div<T>(cell, degree, lvariant,
                                                dvariant, discontinuous);
    default:
      return element::create_bdm<T>(cell, degree, lvariant, discontinuous);
    }
  case element::family::N2E:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_serendipity_curl<T>(cell, degree, lvariant,
                                                 dvariant, discontinuous);
    case cell::type::hexahedron:
      return element::create_serendipity_curl<T>(cell, degree, lvariant,
                                                 dvariant, discontinuous);
    default:
      return element::create_nedelec2<T>(cell, degree, lvariant, discontinuous);
    }
  case element::family::DPC:
    return element::create_dpc<T>(cell, degree, dvariant, discontinuous);

  // Matrix elements
  case element::family::Regge:
    return element::create_regge<T>(cell, degree, discontinuous);
  case element::family::HHJ:
    return element::create_hhj<T>(cell, degree, discontinuous);

  // Other elements
  case element::family::CR:
    return element::create_cr<T>(cell, degree, discontinuous);
  case element::family::bubble:
    return element::create_bubble<T>(cell, degree, discontinuous);
  case element::family::iso:
    return element::create_iso<T>(cell, degree, lvariant, discontinuous);
  case element::family::Hermite:
    return element::create_hermite<T>(cell, degree, discontinuous);
  default:
    throw std::runtime_error("Element family not found.");
  }
}
//-----------------------------------------------------------------------------
template basix::FiniteElement<float>
basix::create_element(element::family, cell::type, int,
                      element::lagrange_variant, element::dpc_variant, bool,
                      std::vector<int>);
template basix::FiniteElement<double>
basix::create_element(element::family, cell::type, int,
                      element::lagrange_variant, element::dpc_variant, bool,
                      std::vector<int>);
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>
basix::create_tp_element(element::family family, cell::type cell, int degree,
                         element::lagrange_variant lvariant,
                         element::dpc_variant dvariant, bool discontinuous)
{
  std::vector<int> dof_ordering = tp_dof_ordering(
      family, cell, degree, lvariant, dvariant, discontinuous);
  return create_element<T>(family, cell, degree, lvariant, dvariant,
                           discontinuous, dof_ordering);
}
//-----------------------------------------------------------------------------
template basix::FiniteElement<float>
basix::create_tp_element(element::family, cell::type, int,
                         element::lagrange_variant, element::dpc_variant, bool);
template basix::FiniteElement<double>
basix::create_tp_element(element::family, cell::type, int,
                         element::lagrange_variant, element::dpc_variant, bool);
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::vector<std::vector<FiniteElement<T>>>
basix::tp_factors(element::family family, cell::type cell, int degree,
                  element::lagrange_variant lvariant,
                  element::dpc_variant dvariant, bool discontinuous,
                  std::vector<int> dof_ordering)
{
  std::vector<int> tp_dofs = tp_dof_ordering(family, cell, degree, lvariant,
                                             dvariant, discontinuous);
  if (!tp_dofs.empty() && tp_dofs == dof_ordering)
  {
    switch (family)
    {
    case element::family::P:
    {
      FiniteElement<T> sub_element
          = create_element<T>(element::family::P, cell::type::interval, degree,
                              lvariant, dvariant, true);
      switch (cell)
      {
      case cell::type::quadrilateral:
      {
        return {{sub_element, sub_element}};
      }
      case cell::type::hexahedron:
      {
        return {{sub_element, sub_element, sub_element}};
      }
      default:
      {
        throw std::runtime_error("Invalid celltype.");
      }
      }
      break;
    }
    default:
    {
      throw std::runtime_error("Invalid family.");
    }
    }
  }
  throw std::runtime_error(
      "Element does not have tensor product factorisation.");
}
//-----------------------------------------------------------------------------
template std::vector<std::vector<basix::FiniteElement<float>>>
basix::tp_factors(element::family, cell::type, int, element::lagrange_variant,
                  element::dpc_variant, bool, std::vector<int>);
template std::vector<std::vector<basix::FiniteElement<double>>>
basix::tp_factors(element::family, cell::type, int, element::lagrange_variant,
                  element::dpc_variant, bool, std::vector<int>);
//-----------------------------------------------------------------------------
std::vector<int> basix::tp_dof_ordering(element::family family, cell::type cell,
                                        int degree, element::lagrange_variant,
                                        element::dpc_variant, bool)
{
  std::vector<int> dof_ordering;
  std::vector<int> perm;

  switch (family)
  {
  case element::family::P:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
    {
      perm.push_back(0);
      if (degree > 0)
      {
        int n = degree - 1;
        perm.push_back(2);
        for (int i = 0; i < n; ++i)
          perm.push_back(4 + n + i);
        perm.push_back(1);
        perm.push_back(3);
        for (int i = 0; i < n; ++i)
          perm.push_back(4 + 2 * n + i);
        for (int i = 0; i < n; ++i)
        {
          perm.push_back(4 + i);
          perm.push_back(4 + 3 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(4 + i + (4 + j) * n);
        }
      }
      assert((int)perm.size() == (degree + 1) * (degree + 1));
      break;
    }
    case cell::type::hexahedron:
    {
      perm.push_back(0);
      if (degree > 0)
      {
        int n = degree - 1;
        perm.push_back(4);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 2 * n + i);
        perm.push_back(2);
        perm.push_back(6);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 6 * n + i);
        for (int i = 0; i < n; ++i)
        {
          perm.push_back(8 + n + i);
          perm.push_back(8 + 9 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + 2 * n * n + i + n * j);
        }
        perm.push_back(1);
        perm.push_back(5);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 4 * n + i);
        perm.push_back(3);
        perm.push_back(7);
        for (int i = 0; i < n; ++i)
          perm.push_back(8 + 7 * n + i);
        for (int i = 0; i < n; ++i)
        {
          perm.push_back(8 + 3 * n + i);
          perm.push_back(8 + 10 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + 3 * n * n + i + n * j);
        }
        for (int i = 0; i < n; ++i)
        {
          perm.push_back(8 + i);
          perm.push_back(8 + 8 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + n * n + i + n * j);
          perm.push_back(8 + 5 * n + i);
          perm.push_back(8 + 11 * n + i);
          for (int j = 0; j < n; ++j)
            perm.push_back(8 + 12 * n + 4 * n * n + i + n * j);
          for (int j = 0; j < n; ++j)
          {
            perm.push_back(8 + 12 * n + i + n * j);
            perm.push_back(8 + 12 * n + 5 * n * n + i + n * j);
            for (int k = 0; k < n; ++k)
              perm.push_back(8 + 12 * n + 6 * n * n + i + n * j + n * n * k);
          }
        }
      }
      assert((int)perm.size() == (degree + 1) * (degree + 1) * (degree + 1));
      break;
    }
    default:
    {
    }
    }
    break;
  }
  default:
  {
  }
  }

  if (perm.size() == 0)
  {
    throw std::runtime_error(
        "Element does not have tensor product factorisation.");
  }
  dof_ordering.resize(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i)
    dof_ordering[perm[i]] = i;
  return dof_ordering;
}
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
          new_x[k0 + x_n, k1] = x[i][j][k0, k1];

      for (std::size_t k0 = 0; k0 < M[i][j].extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < M[i][j].extent(1); ++k1)
          for (std::size_t k2 = 0; k2 < M[i][j].extent(2); ++k2)
            for (std::size_t k3 = 0; k3 < M[i][j].extent(3); ++k3)
              new_M[k0 + M_n, k1, k2 + x_n, k3] = M[i][j][k0, k1, k2, k3];

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
template <std::floating_point T>
FiniteElement<T> basix::create_custom_element(
    cell::type cell_type, const std::vector<std::size_t>& value_shape,
    impl::mdspan_t<const T, 2> wcoeffs,
    const std::array<std::vector<impl::mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<impl::mdspan_t<const T, 4>>, 4>& M,
    int interpolation_nderivs, maps::type map_type,
    sobolev::space sobolev_space, bool discontinuous, int embedded_subdegree,
    int embedded_superdegree, polyset::type poly_type)
{
  // Check that inputs are valid
  const std::size_t psize
      = polyset::dim(cell_type, poly_type, embedded_superdegree);
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

  std::vector<T> wcoeffs_ortho_b(wcoeffs.extent(0) * wcoeffs.extent(1));
  { // scope
    mdspan_t<T, 2> wcoeffs_ortho(wcoeffs_ortho_b.data(), wcoeffs.extent(0),
                                 wcoeffs.extent(1));
    std::copy(wcoeffs.data_handle(), wcoeffs.data_handle() + wcoeffs.size(),
              wcoeffs_ortho_b.begin());
    basix::math::orthogonalise(wcoeffs_ortho);
  }
  mdspan_t<const T, 2> wcoeffs_ortho(wcoeffs_ortho_b.data(), wcoeffs.extent(0),
                                     wcoeffs.extent(1));

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

  auto [dualmatrix, dualshape]
      = compute_dual_matrix(cell_type, poly_type, wcoeffs_ortho, x, M,
                            embedded_superdegree, interpolation_nderivs);
  if (math::is_singular(mdspan_t<const T, 2>(dualmatrix.data(), dualshape)))
  {
    throw std::runtime_error(
        "Dual matrix is singular, there is an error in your inputs");
  }

  return basix::FiniteElement<T>(
      element::family::custom, cell_type, poly_type, embedded_superdegree,
      value_shape, wcoeffs_ortho, x, M, interpolation_nderivs, map_type,
      sobolev_space, discontinuous, embedded_subdegree, embedded_superdegree,
      element::lagrange_variant::unset, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
/// @cond
template FiniteElement<float> basix::create_custom_element(
    cell::type, const std::vector<std::size_t>&,
    impl::mdspan_t<const float, 2> wcoeffs,
    const std::array<std::vector<impl::mdspan_t<const float, 2>>, 4>&,
    const std::array<std::vector<impl::mdspan_t<const float, 4>>, 4>&, int,
    maps::type, sobolev::space sobolev_space, bool, int, int, polyset::type);
template FiniteElement<double> basix::create_custom_element(
    cell::type, const std::vector<std::size_t>&,
    impl::mdspan_t<const double, 2> wcoeffs,
    const std::array<std::vector<impl::mdspan_t<const double, 2>>, 4>&,
    const std::array<std::vector<impl::mdspan_t<const double, 4>>, 4>&, int,
    maps::type, sobolev::space sobolev_space, bool, int, int, polyset::type);
/// @endcond
//-----------------------------------------------------------------------------
/// @cond
template <std::floating_point F>
FiniteElement<F>::FiniteElement(
    element::family family, cell::type cell_type, polyset::type poly_type,
    int degree, const std::vector<std::size_t>& value_shape,
    mdspan_t<const F, 2> wcoeffs,
    const std::array<std::vector<mdspan_t<const F, 2>>, 4>& x,
    const std::array<std::vector<mdspan_t<const F, 4>>, 4>& M,
    int interpolation_nderivs, maps::type map_type,
    sobolev::space sobolev_space, bool discontinuous, int embedded_subdegree,
    int embedded_superdegree, element::lagrange_variant lvariant,
    element::dpc_variant dvariant, std::vector<int> dof_ordering)
    : _cell_type(cell_type), _poly_type(poly_type),
      _cell_tdim(cell::topological_dimension(cell_type)),
      _cell_subentity_types(cell::subentity_types(cell_type)), _family(family),
      _lagrange_variant(lvariant), _dpc_variant(dvariant), _degree(degree),
      _interpolation_nderivs(interpolation_nderivs),
      _embedded_superdegree(embedded_superdegree),
      _embedded_subdegree(embedded_subdegree), _value_shape(value_shape),
      _map_type(map_type), _sobolev_space(sobolev_space),
      _discontinuous(discontinuous), _dof_ordering(dof_ordering)
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

  try
  {
    _tensor_factors = tp_factors<F>(family, cell_type, degree, lvariant,
                                    dvariant, discontinuous, dof_ordering);
  }
  catch (...)
  {
  }

  std::vector<F> wcoeffs_b(wcoeffs.extent(0) * wcoeffs.extent(1));
  std::copy(wcoeffs.data_handle(), wcoeffs.data_handle() + wcoeffs.size(),
            wcoeffs_b.begin());

  _wcoeffs = {wcoeffs_b, {wcoeffs.extent(0), wcoeffs.extent(1)}};
  _dual_matrix
      = compute_dual_matrix<F>(cell_type, poly_type, wcoeffs, x, M,
                               embedded_superdegree, interpolation_nderivs);

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

  // Copy M
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
  _coeffs.first = math::solve<F>(
      mdspan_t<const F, 2>(_dual_matrix.first.data(), _dual_matrix.second),
      wcoeffs);
  _coeffs.second = {_dual_matrix.second[1], wcoeffs.extent(1)};

  std::size_t num_points = 0;
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      num_points += x_e.extent(0);

  _points.first.reserve(num_points * _cell_tdim);
  _points.second = {num_points, _cell_tdim};
  mdspan_t<F, 2> pview(_points.first.data(), _points.second);
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      for (std::size_t p = 0; p < x_e.extent(0); ++p)
        for (std::size_t k = 0; k < x_e.extent(1); ++k)
          _points.first.push_back(x_e[p, k]);

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
      cell_type, x, M,
      mdspan_t<const F, 2>(_coeffs.first.data(), _coeffs.second),
      embedded_superdegree, value_size, map_type, poly_type);

  const std::size_t nderivs
      = polyset::nderivs(cell_type, interpolation_nderivs);

  _matM = {std::vector<F>(num_dofs * value_size * num_points1 * nderivs),
           {num_dofs, value_size * num_points1 * nderivs}};
  mdspan_t<F, 4> Mview(_matM.first.data(), num_dofs, value_size, num_points1,
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
              Mview[k0 + dof_offset, k1, k2 + point_offset, k3]
                  = Me[k0, k1, k2, k3];

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

  if (!_dof_ordering.empty())
  {
    const int ndof_order = _dof_ordering.size();
    // Safety checks
    if (ndof_order != dof)
      throw std::runtime_error("Incorrect number of dofs in ordering.");
    std::vector<int> check(_dof_ordering.size(), 0);
    for (int q : _dof_ordering)
    {
      if (q < 0 or q >= ndof_order)
        throw std::runtime_error("Out of range: dof_ordering.");
      check[q] += 1;
    }
    for (int q : check)
      if (q != 1)
        throw std::runtime_error("Dof ordering not a permutation.");

    // Apply permutation to _edofs
    for (std::size_t d = 0; d < _cell_tdim + 1; ++d)
    {
      for (auto& entity : _edofs[d])
      {
        for (int& q : entity)
          q = _dof_ordering[q];
      }
    }

    // Apply permutation to _points (for interpolation)
    std::vector<F> new_points(_points.first.size());
    assert(_points.second[0] == _dof_ordering.size());
    const int gdim = _points.second[1];
    for (std::size_t d = 0; d < _dof_ordering.size(); ++d)
      for (int i = 0; i < gdim; ++i)
        new_points[gdim * _dof_ordering[d] + i] = _points.first[gdim * d + i];
    _points = {new_points, _points.second};
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

      std::ranges::sort(_e_closure_dofs[d][e]);
    }
  }

  // Check if base transformations are all permutations
  _dof_transformations_are_permutations = true;
  _dof_transformations_are_identity = true;
  for (const auto& [ctype, trans_data] : _entity_transformations)
  {
    mdspan_t<const F, 3> trans(trans_data.first.data(), trans_data.second);
    for (std::size_t i = 0;
         _dof_transformations_are_permutations and i < trans.extent(0); ++i)
    {
      for (std::size_t row = 0; row < trans.extent(1); ++row)
      {
        F rmin(0), rmax(0), rtot(0);
        for (std::size_t k = 0; k < trans.extent(2); ++k)
        {
          F r = trans[i, row, k];
          rmin = std::min(r, rmin);
          rmax = std::max(r, rmax);
          rtot += r;
        }

        constexpr F eps = 10.0 * std::numeric_limits<float>::epsilon();
        if ((trans.extent(2) != 1 and std::abs(rmin) > eps)
            or std::abs(rmax - 1.0) > eps or std::abs(rtot - 1.0) > eps)
        {
          _dof_transformations_are_permutations = false;
          _dof_transformations_are_identity = false;
          break;
        }

        if (std::abs(trans[i, row, row] - 1) > eps)
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
        mdspan_t<const F, 3> trans(trans_data.first.data(), trans_data.second);
        for (std::size_t i = 0; i < trans.extent(0); ++i)
        {
          std::vector<std::size_t> perm(trans.extent(1));
          std::vector<std::size_t> inv_perm(trans.extent(1));
          for (std::size_t row = 0; row < trans.extent(1); ++row)
          {
            for (std::size_t col = 0; col < trans.extent(1); ++col)
            {
              if (trans[i, row, col] > 0.5)
              {
                perm[row] = col;
                inv_perm[col] = row;
                break;
              }
            }
          }

          // Factorise the permutations
          precompute::prepare_permutation(perm);
          precompute::prepare_permutation(inv_perm);

          // Store the permutations
          auto& eperm = _eperm.try_emplace(ctype).first->second;
          auto& eperm_inv = _eperm_inv.try_emplace(ctype).first->second;
          eperm.push_back(perm);
          eperm_inv.push_back(inv_perm);

          // Generate the entity transformations from the permutations
          std::pair<std::vector<F>, std::array<std::size_t, 2>> identity
              = {std::vector<F>(perm.size() * perm.size()),
                 {perm.size(), perm.size()}};
          std::ranges::fill(identity.first, 0.);
          for (std::size_t i = 0; i < perm.size(); ++i)
            identity.first[i * perm.size() + i] = 1;

          auto& etrans = _etrans.try_emplace(ctype).first->second;
          auto& etransT = _etransT.try_emplace(ctype).first->second;
          auto& etrans_invT = _etrans_invT.try_emplace(ctype).first->second;
          auto& etrans_inv = _etrans_inv.try_emplace(ctype).first->second;
          etrans.push_back({perm, identity});
          etrans_invT.push_back({perm, identity});
          etransT.push_back({inv_perm, identity});
          etrans_inv.push_back({inv_perm, identity});
        }
      }
    }
    else
    {
      // Precompute the DOF transformations
      for (const auto& [ctype, trans_data] : _entity_transformations)
      {
        mdspan_t<const F, 3> trans(trans_data.first.data(), trans_data.second);

        // Buffers for matrices
        std::vector<F> M_b, Minv_b, matint;
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
              std::pair<std::vector<F>, std::array<std::size_t, 2>> mat
                  = {std::vector<F>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  mat.first[k0 * dim + k1] = trans[i, k0, k1];
              std::vector<std::size_t> mat_p = precompute::prepare_matrix(mat);
              etrans.push_back({mat_p, mat});
            }

            {
              std::pair<std::vector<F>, std::array<std::size_t, 2>> matT
                  = {std::vector<F>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  matT.first[k0 * dim + k1] = trans[i, k1, k0];
              std::vector<std::size_t> matT_p
                  = precompute::prepare_matrix(matT);
              etransT.push_back({matT_p, matT});
            }

            M_b.resize(dim * dim);
            mdspan_t<F, 2> M(M_b.data(), dim, dim);
            for (std::size_t k0 = 0; k0 < dim; ++k0)
              for (std::size_t k1 = 0; k1 < dim; ++k1)
                M[k0, k1] = trans[i, k0, k1];

            // Rotation of a face: this is in the only base transformation
            // such that M^{-1} != M.
            // For a quadrilateral face, M^4 = Id, so M^{-1} = M^3.
            // For a triangular face, M^3 = Id, so M^{-1} = M^2.
            Minv_b.resize(dim * dim);
            mdspan_t<F, 2> Minv(Minv_b.data(), dim, dim);
            if (ctype == cell::type::quadrilateral and i == 0)
            {
              matint.resize(dim * dim);
              mdspan_t<F, 2> mat_int(matint.data(), dim, dim);
              math::dot(M, M, mat_int);
              math::dot(mat_int, M, Minv);
            }
            else if (ctype == cell::type::triangle and i == 0)
              math::dot(M, M, Minv);
            else
              Minv_b.assign(M_b.begin(), M_b.end());

            {
              std::pair<std::vector<F>, std::array<std::size_t, 2>> mat_inv
                  = {std::vector<F>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  mat_inv.first[k0 * dim + k1] = Minv[k0, k1];
              std::vector<std::size_t> mat_inv_p
                  = precompute::prepare_matrix(mat_inv);
              etrans_inv.push_back({mat_inv_p, mat_inv});
            }

            {
              std::pair<std::vector<F>, std::array<std::size_t, 2>> mat_invT
                  = {std::vector<F>(dim * dim), {dim, dim}};
              for (std::size_t k0 = 0; k0 < dim; ++k0)
                for (std::size_t k1 = 0; k1 < dim; ++k1)
                  mat_invT.first[k0 * dim + k1] = Minv[k1, k0];
              std::vector<std::size_t> mat_invT_p
                  = precompute::prepare_matrix(mat_invT);
              etrans_invT.push_back({mat_invT_p, mat_invT});
            }
          }
        }
      }
    }
  }

  // If DOF transformations are permutations, compute the subentity closure
  // permutations
  if (_dof_transformations_are_permutations)
  {
    if (_cell_tdim > 1)
    {
      // interval
      int dof_n = 0;
      const auto conn = cell::sub_entity_connectivity(_cell_type)[1][0];
      std::vector<std::vector<std::vector<int>>> dofs;
      dofs.resize(2);
      for (int dim = 0; dim <= 1; ++dim)
      {
        dofs[dim].resize(conn[dim].size());
        for (std::size_t i = 0; i < conn[dim].size(); ++i)
        {
          const int e = conn[dim][i];
          for (std::size_t j = 0; j < _edofs[dim][e].size(); ++j)
          {
            dofs[dim][i].push_back(dof_n++);
          }
        }
      }

      std::vector<std::size_t> ref;
      ref.insert(ref.end(), dofs[0][1].begin(), dofs[0][1].end());
      ref.insert(ref.end(), dofs[0][0].begin(), dofs[0][0].end());
      // Edges
      ref.insert(ref.end(), dofs[1][0].begin(), dofs[1][0].end());

      if (!_dof_transformations_are_identity)
      {
        auto& trans1 = _eperm.at(cell::type::interval)[0];
        if (!dofs[1][0].empty())
        {
          precompute::apply_permutation(trans1, std::span(ref), dofs[1][0][0]);
        }
      }

      precompute::prepare_permutation(ref);

      auto& secp = _subentity_closure_perm.try_emplace(cell::type::interval)
                       .first->second;
      secp.push_back(ref);
      auto& secpi
          = _subentity_closure_perm_inv.try_emplace(cell::type::interval)
                .first->second;
      secpi.push_back(ref);
    }
    if (_cell_type == cell::type::tetrahedron || cell_type == cell::type::prism
        || cell_type == cell::type::pyramid)
    {
      // triangle
      const int face_n = cell_type == cell::type::pyramid ? 1 : 0;

      int dof_n = 0;
      const auto conn = cell::sub_entity_connectivity(_cell_type)[2][face_n];
      std::vector<std::vector<std::vector<int>>> dofs;
      dofs.resize(3);
      for (int dim = 0; dim <= 2; ++dim)
      {
        dofs[dim].resize(conn[dim].size());
        for (std::size_t i = 0; i < conn[dim].size(); ++i)
        {
          const int e = conn[dim][i];
          for (int j : _edofs[dim][e])
          {
            std::ignore = j;
            dofs[dim][i].push_back(dof_n++);
          }
        }
      }

      std::vector<std::size_t> rot;
      // Vertices
      rot.insert(rot.end(), dofs[0][1].begin(), dofs[0][1].end());
      rot.insert(rot.end(), dofs[0][2].begin(), dofs[0][2].end());
      rot.insert(rot.end(), dofs[0][0].begin(), dofs[0][0].end());
      // Edges
      rot.insert(rot.end(), dofs[1][1].begin(), dofs[1][1].end());
      rot.insert(rot.end(), dofs[1][2].begin(), dofs[1][2].end());
      rot.insert(rot.end(), dofs[1][0].begin(), dofs[1][0].end());
      // Face
      rot.insert(rot.end(), dofs[2][0].begin(), dofs[2][0].end());

      std::vector<std::size_t> rot_inv;
      // Vertices
      rot_inv.insert(rot_inv.end(), dofs[0][2].begin(), dofs[0][2].end());
      rot_inv.insert(rot_inv.end(), dofs[0][0].begin(), dofs[0][0].end());
      rot_inv.insert(rot_inv.end(), dofs[0][1].begin(), dofs[0][1].end());
      // Edges
      rot_inv.insert(rot_inv.end(), dofs[1][2].begin(), dofs[1][2].end());
      rot_inv.insert(rot_inv.end(), dofs[1][0].begin(), dofs[1][0].end());
      rot_inv.insert(rot_inv.end(), dofs[1][1].begin(), dofs[1][1].end());
      // Face
      rot_inv.insert(rot_inv.end(), dofs[2][0].begin(), dofs[2][0].end());

      std::vector<std::size_t> ref;
      // Vertices
      ref.insert(ref.end(), dofs[0][0].begin(), dofs[0][0].end());
      ref.insert(ref.end(), dofs[0][2].begin(), dofs[0][2].end());
      ref.insert(ref.end(), dofs[0][1].begin(), dofs[0][1].end());
      // Edges
      ref.insert(ref.end(), dofs[1][0].begin(), dofs[1][0].end());
      ref.insert(ref.end(), dofs[1][2].begin(), dofs[1][2].end());
      ref.insert(ref.end(), dofs[1][1].begin(), dofs[1][1].end());
      // Face
      ref.insert(ref.end(), dofs[2][0].begin(), dofs[2][0].end());

      if (!_dof_transformations_are_identity)
      {
        auto& trans1 = _eperm.at(cell::type::interval)[0];
        auto& trans2 = _eperm.at(cell::type::triangle);
        auto& trans3 = _eperm_inv.at(cell::type::triangle);

        if (!dofs[1][0].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot), dofs[1][0][0]);
        }
        if (!dofs[1][1].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot), dofs[1][1][0]);
        }
        if (!dofs[2][0].empty())
        {
          precompute::apply_permutation(trans2[0], std::span(rot),
                                        dofs[2][0][0]);
        }
        if (!dofs[1][0].empty())
        {
          precompute::apply_permutation(trans1, std::span(ref), dofs[1][0][0]);
        }
        if (!dofs[2][0].empty())
        {
          precompute::apply_permutation(trans2[1], std::span(ref),
                                        dofs[2][0][0]);
        }
        if (!dofs[1][1].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot_inv),
                                        dofs[1][1][0]);
        }
        if (!dofs[1][2].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot_inv),
                                        dofs[1][2][0]);
        }
        if (!dofs[2][0].empty())
        {
          precompute::apply_permutation(trans3[0], std::span(rot_inv),
                                        dofs[2][0][0]);
        }
      }

      precompute::prepare_permutation(rot);
      precompute::prepare_permutation(rot_inv);
      precompute::prepare_permutation(ref);

      auto& secp = _subentity_closure_perm.try_emplace(cell::type::triangle)
                       .first->second;
      secp.push_back(rot);
      secp.push_back(ref);
      auto& secpi
          = _subentity_closure_perm_inv.try_emplace(cell::type::triangle)
                .first->second;
      secpi.push_back(rot_inv);
      secpi.push_back(ref);
    }
    if (_cell_type == cell::type::hexahedron || cell_type == cell::type::prism
        || cell_type == cell::type::pyramid)
    {
      // quadrilateral
      const int face_n = cell_type == cell::type::prism ? 1 : 0;

      int dof_n = 0;
      const auto conn = cell::sub_entity_connectivity(_cell_type)[2][face_n];
      std::vector<std::vector<std::vector<int>>> dofs;
      dofs.resize(3);
      for (int dim = 0; dim <= 2; ++dim)
      {
        dofs[dim].resize(conn[dim].size());
        for (std::size_t i = 0; i < conn[dim].size(); ++i)
        {
          const int e = conn[dim][i];
          for (int j : _edofs[dim][e])
          {
            std::ignore = j;
            dofs[dim][i].push_back(dof_n++);
          }
        }
      }

      std::vector<std::size_t> rot;
      // Vertices
      rot.insert(rot.end(), dofs[0][1].begin(), dofs[0][1].end());
      rot.insert(rot.end(), dofs[0][3].begin(), dofs[0][3].end());
      rot.insert(rot.end(), dofs[0][0].begin(), dofs[0][0].end());
      rot.insert(rot.end(), dofs[0][2].begin(), dofs[0][2].end());
      // Edges
      rot.insert(rot.end(), dofs[1][2].begin(), dofs[1][2].end());
      rot.insert(rot.end(), dofs[1][0].begin(), dofs[1][0].end());
      rot.insert(rot.end(), dofs[1][3].begin(), dofs[1][3].end());
      rot.insert(rot.end(), dofs[1][1].begin(), dofs[1][1].end());
      // Face
      rot.insert(rot.end(), dofs[2][0].begin(), dofs[2][0].end());

      std::vector<std::size_t> rot_inv;
      // Vertices
      rot_inv.insert(rot_inv.end(), dofs[0][2].begin(), dofs[0][2].end());
      rot_inv.insert(rot_inv.end(), dofs[0][0].begin(), dofs[0][0].end());
      rot_inv.insert(rot_inv.end(), dofs[0][3].begin(), dofs[0][3].end());
      rot_inv.insert(rot_inv.end(), dofs[0][1].begin(), dofs[0][1].end());
      // Edges
      rot_inv.insert(rot_inv.end(), dofs[1][1].begin(), dofs[1][1].end());
      rot_inv.insert(rot_inv.end(), dofs[1][3].begin(), dofs[1][3].end());
      rot_inv.insert(rot_inv.end(), dofs[1][0].begin(), dofs[1][0].end());
      rot_inv.insert(rot_inv.end(), dofs[1][2].begin(), dofs[1][2].end());
      // Face
      rot_inv.insert(rot_inv.end(), dofs[2][0].begin(), dofs[2][0].end());

      std::vector<std::size_t> ref;
      ref.insert(ref.end(), dofs[0][0].begin(), dofs[0][0].end());
      ref.insert(ref.end(), dofs[0][2].begin(), dofs[0][2].end());
      ref.insert(ref.end(), dofs[0][1].begin(), dofs[0][1].end());
      ref.insert(ref.end(), dofs[0][3].begin(), dofs[0][3].end());
      // Edges
      ref.insert(ref.end(), dofs[1][1].begin(), dofs[1][1].end());
      ref.insert(ref.end(), dofs[1][0].begin(), dofs[1][0].end());
      ref.insert(ref.end(), dofs[1][3].begin(), dofs[1][3].end());
      ref.insert(ref.end(), dofs[1][2].begin(), dofs[1][2].end());
      // Face
      ref.insert(ref.end(), dofs[2][0].begin(), dofs[2][0].end());

      if (!_dof_transformations_are_identity)
      {
        auto& trans1 = _eperm.at(cell::type::interval)[0];
        auto& trans2 = _eperm.at(cell::type::quadrilateral);
        auto& trans3 = _eperm_inv.at(cell::type::quadrilateral);

        if (!dofs[1][1].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot), dofs[1][1][0]);
        }
        if (!dofs[1][2].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot), dofs[1][2][0]);
        }
        if (!dofs[2][0].empty())
        {
          precompute::apply_permutation(trans2[0], std::span(rot),
                                        dofs[2][0][0]);
        }

        if (!dofs[2][0].empty())
        {
          precompute::apply_permutation(trans2[1], std::span(ref),
                                        dofs[2][0][0]);
        }

        if (!dofs[1][1].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot_inv),
                                        dofs[1][1][0]);
        }
        if (!dofs[1][2].empty())
        {
          precompute::apply_permutation(trans1, std::span(rot_inv),
                                        dofs[1][2][0]);
        }
        if (!dofs[2][0].empty())
        {
          precompute::apply_permutation(trans3[0], std::span(rot_inv),
                                        dofs[2][0][0]);
        }
      }

      precompute::prepare_permutation(rot);
      precompute::prepare_permutation(rot_inv);
      precompute::prepare_permutation(ref);

      auto& secp
          = _subentity_closure_perm.try_emplace(cell::type::quadrilateral)
                .first->second;
      secp.push_back(rot);
      secp.push_back(ref);
      auto& secpi
          = _subentity_closure_perm_inv.try_emplace(cell::type::quadrilateral)
                .first->second;
      secpi.push_back(rot_inv);
      secpi.push_back(ref);
    }
  }

  // Check if interpolation matrix is the identity
  mdspan_t<const F, 2> matM(_matM.first.data(), _matM.second);
  _interpolation_is_identity = matM.extent(0) == matM.extent(1);
  for (std::size_t row = 0; _interpolation_is_identity && row < matM.extent(0);
       ++row)
  {
    for (std::size_t col = 0; col < matM.extent(1); ++col)
    {
      F v = col == row ? 1.0 : 0.0;
      constexpr F eps = 100 * std::numeric_limits<F>::epsilon();
      if (std::abs(matM[row, col] - v) > eps)
      {
        _interpolation_is_identity = false;
        break;
      }
    }
  }
}
/// @endcond
//-----------------------------------------------------------------------------
template <std::floating_point F>
bool FiniteElement<F>::operator==(const FiniteElement& e) const
{
  if (this == &e)
    return true;
  else if (family() == element::family::custom
           and e.family() == element::family::custom)
  {
    bool coeff_equal = false;
    if (_coeffs.first.size() == e.coefficient_matrix().first.size()
        and _coeffs.second == e.coefficient_matrix().second
        and std::ranges::equal(_coeffs.first, e.coefficient_matrix().first,
                               [](auto x, auto y)
                               { return std::abs(x - y) < 1.0e-10; }))
    {
      coeff_equal = true;
    }

    return cell_type() == e.cell_type() and discontinuous() == e.discontinuous()
           and map_type() == e.map_type()
           and sobolev_space() == e.sobolev_space()
           and value_shape() == e.value_shape()
           and embedded_superdegree() == e.embedded_superdegree()
           and embedded_subdegree() == e.embedded_subdegree() and coeff_equal
           and entity_dofs() == e.entity_dofs()
           and dof_ordering() == e.dof_ordering()
           and polyset_type() == e.polyset_type();
  }
  else
  {
    return cell_type() == e.cell_type() and family() == e.family()
           and degree() == e.degree() and discontinuous() == e.discontinuous()
           and lagrange_variant() == e.lagrange_variant()
           and dpc_variant() == e.dpc_variant() and map_type() == e.map_type()
           and sobolev_space() == e.sobolev_space()
           and dof_ordering() == e.dof_ordering();
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
std::size_t FiniteElement<F>::hash() const
{
  std::size_t dof_ordering_hash = 0;
  for (std::size_t i = 0; i < dof_ordering().size(); ++i)
  {
    if (dof_ordering()[i] != static_cast<int>(i))
    {
      combine_hashes(dof_ordering_hash,
                     std::hash<int>{}(dof_ordering()[i] - i));
    }
  }

  std::size_t h = std::hash<int>{}(static_cast<int>(family()));
  combine_hashes(h, dof_ordering_hash);
  combine_hashes(h, dof_ordering_hash);
  combine_hashes(h, std::hash<int>{}(static_cast<int>(cell_type())));
  combine_hashes(h, std::hash<int>{}(static_cast<int>(lagrange_variant())));
  combine_hashes(h, std::hash<int>{}(static_cast<int>(dpc_variant())));
  combine_hashes(h, std::hash<int>{}(static_cast<int>(sobolev_space())));
  combine_hashes(h, std::hash<int>{}(static_cast<int>(map_type())));

  if (family() == element::family::custom)
  {
    std::size_t coeff_hash = 0;
    for (auto i : _coeffs.first)
    {
      // This takes five decimal places of each matrix entry. We should revisit
      // this
      combine_hashes(coeff_hash, int(i * 100000));
    }
    std::size_t vs_hash = 0;
    for (std::size_t i = 0; i < value_shape().size(); ++i)
    {
      combine_hashes(vs_hash, std::hash<int>{}(value_shape()[i]));
    }
    combine_hashes(h, coeff_hash);
    combine_hashes(h, std::hash<int>{}(embedded_superdegree()));
    combine_hashes(h, std::hash<int>{}(embedded_subdegree()));
    combine_hashes(h, std::hash<int>{}(static_cast<int>(polyset_type())));
    combine_hashes(h, vs_hash);
  }
  else
  {
    combine_hashes(h, std::hash<int>{}(degree()));
  }
  return h;
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
std::pair<std::vector<F>, std::array<std::size_t, 4>>
FiniteElement<F>::tabulate(int nd, impl::mdspan_t<const F, 2> x) const
{
  std::array<std::size_t, 4> shape = tabulate_shape(nd, x.extent(0));
  std::vector<F> data(shape[0] * shape[1] * shape[2] * shape[3]);
  tabulate(nd, x, mdspan_t<F, 4>(data.data(), shape));
  return {std::move(data), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
std::pair<std::vector<F>, std::array<std::size_t, 4>>
FiniteElement<F>::tabulate(int nd, std::span<const F> x,
                           std::array<std::size_t, 2> shape) const
{
  std::array<std::size_t, 4> phishape = tabulate_shape(nd, shape[0]);
  std::vector<F> datab(phishape[0] * phishape[1] * phishape[2] * phishape[3]);
  tabulate(nd, mdspan_t<const F, 2>(x.data(), shape[0], shape[1]),
           mdspan_t<F, 4>(datab.data(), phishape));
  return {std::move(datab), phishape};
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
void FiniteElement<F>::tabulate(int nd, impl::mdspan_t<const F, 2> x,
                                mdspan_t<F, 4> basis_data) const
{
  if (x.extent(1) != _cell_tdim)
  {
    throw std::runtime_error("Point dim (" + std::to_string(x.extent(1))
                             + ") does not match element dim ("
                             + std::to_string(_cell_tdim) + ").");
  }

  const std::size_t psize
      = polyset::dim(_cell_type, _poly_type, _embedded_superdegree);
  const std::array<std::size_t, 3> bsize
      = {(std::size_t)polyset::nderivs(_cell_type, nd), psize, x.extent(0)};
  std::vector<F> basis_b(bsize[0] * bsize[1] * bsize[2]);
  mdspan_t<F, 3> basis(basis_b.data(), bsize);
  polyset::tabulate(basis, _cell_type, _poly_type, _embedded_superdegree, nd,
                    x);
  const int vs = std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                                 std::multiplies{});

  std::vector<F> C_b(_coeffs.second[0] * psize);
  mdspan_t<F, 2> C(C_b.data(), _coeffs.second[0], psize);

  mdspan_t<const F, 2> coeffs_view(_coeffs.first.data(), _coeffs.second);
  std::vector<F> result_b(C.extent(0) * bsize[2]);
  mdspan_t<F, 2> result(result_b.data(), C.extent(0), bsize[2]);
  for (std::size_t p = 0; p < basis.extent(0); ++p)
  {
    mdspan_t<const F, 2> B(basis_b.data() + p * bsize[1] * bsize[2], bsize[1],
                           bsize[2]);
    for (int j = 0; j < vs; ++j)
    {
      for (std::size_t k0 = 0; k0 < coeffs_view.extent(0); ++k0)
        for (std::size_t k1 = 0; k1 < psize; ++k1)
          C[k0, k1] = coeffs_view[k0, k1 + psize * j];

      math::dot(C,
                mdspan_t<const F, 2>(B.data_handle(), B.extent(0), B.extent(1)),
                result);

      if (_dof_ordering.empty())
      {
        for (std::size_t k0 = 0; k0 < basis_data.extent(1); ++k0)
          for (std::size_t k1 = 0; k1 < basis_data.extent(2); ++k1)
            basis_data[p, k0, k1, j] = result[k1, k0];
      }
      else
      {
        for (std::size_t k0 = 0; k0 < basis_data.extent(1); ++k0)
          for (std::size_t k1 = 0; k1 < basis_data.extent(2); ++k1)
            basis_data[p, k0, _dof_ordering[k1], j] = result[k1, k0];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
void FiniteElement<F>::tabulate(int nd, std::span<const F> x,
                                std::array<std::size_t, 2> xshape,
                                std::span<F> basis) const
{
  std::array<std::size_t, 4> shape = tabulate_shape(nd, xshape[0]);
  assert(x.size() == xshape[0] * xshape[1]);
  assert(basis.size() == shape[0] * shape[1] * shape[2] * shape[3]);
  tabulate(nd, mdspan_t<const F, 2>(x.data(), xshape),
           mdspan_t<F, 4>(basis.data(), shape));
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
std::pair<std::vector<F>, std::array<std::size_t, 3>>
FiniteElement<F>::base_transformations() const
{
  const std::size_t nt = num_transformations(this->cell_type());
  const std::size_t ndofs = this->dim();

  std::array<std::size_t, 3> shape = {nt, ndofs, ndofs};
  std::vector<F> bt_b(shape[0] * shape[1] * shape[2], 0);
  mdspan_t<F, 3> bt(bt_b.data(), shape);
  for (std::size_t i = 0; i < nt; ++i)
    for (std::size_t j = 0; j < ndofs; ++j)
      bt[i, j, j] = 1.0;

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
      mdspan_t<const F, 3> tmp(tmp_data.first.data(), tmp_data.second);
      for (auto& e : _edofs[1])
      {
        std::size_t ndofs = e.size();
        for (std::size_t i = 0; i < ndofs; ++i)
          for (std::size_t j = 0; j < ndofs; ++j)
            bt[transform_n, i + dofstart, j + dofstart] = tmp[0, i, j];

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
          mdspan_t<const F, 3> tmp(tmp_data.first.data(), tmp_data.second);

          for (std::size_t i = 0; i < ndofs; ++i)
            for (std::size_t j = 0; j < ndofs; ++j)
              bt[transform_n, i + dofstart, j + dofstart] = tmp[0, i, j];
          ++transform_n;

          for (std::size_t i = 0; i < ndofs; ++i)
            for (std::size_t j = 0; j < ndofs; ++j)
              bt[transform_n, i + dofstart, j + dofstart] = tmp[1, i, j];
          ++transform_n;

          dofstart += ndofs;
        }
      }
    }
  }

  return {std::move(bt_b), shape};
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
std::pair<std::vector<F>, std::array<std::size_t, 3>>
FiniteElement<F>::push_forward(impl::mdspan_t<const F, 3> U,
                               impl::mdspan_t<const F, 3> J,
                               std::span<const F> detJ,
                               impl::mdspan_t<const F, 3> K) const
{
  const std::size_t physical_value_size
      = compute_value_size(_map_type, J.extent(1));

  std::array<std::size_t, 3> shape
      = {U.extent(0), U.extent(1), physical_value_size};
  std::vector<F> ub(shape[0] * shape[1] * shape[2]);
  mdspan_t<F, 3> u(ub.data(), shape);

  using u_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using U_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using J_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using K_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
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
template <std::floating_point F>
std::pair<std::vector<F>, std::array<std::size_t, 3>>
FiniteElement<F>::pull_back(impl::mdspan_t<const F, 3> u,
                            impl::mdspan_t<const F, 3> J,
                            std::span<const F> detJ,
                            impl::mdspan_t<const F, 3> K) const
{
  const std::size_t reference_value_size = std::accumulate(
      _value_shape.begin(), _value_shape.end(), 1, std::multiplies{});

  std::array<std::size_t, 3> shape
      = {u.extent(0), u.extent(1), reference_value_size};
  std::vector<F> Ub(shape[0] * shape[1] * shape[2]);
  mdspan_t<F, 3> U(Ub.data(), shape);

  using u_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using U_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using J_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using K_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const F, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
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
std::string basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
/// @cond
template class basix::FiniteElement<float>;
template class basix::FiniteElement<double>;
/// @endcond
//-----------------------------------------------------------------------------
