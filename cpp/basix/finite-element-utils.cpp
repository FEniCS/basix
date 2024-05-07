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

template <typename T, std::size_t d>
using mdspan_t = impl::mdspan_t<T, d>;
template <typename T, std::size_t d>
using mdarray_t = impl::mdarray_t<T, d>;

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
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
element::compute_dual_matrix(
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
                  D(j, m, dof_index + i) += Me(i, j, k, l) * P(l, m, k);
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
            Pt(i, j) = P(0, j, i);

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
              D(j, k, dof_index + i) += De(i * Me.extent(1) + j, k);
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

  if (dof_ordering.size() > 0 and family != element::family::P)
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
  if (tp_dofs.size() > 0 && tp_dofs == dof_ordering)
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

  auto [dualmatrix, dualshape] = element::compute_dual_matrix(
      cell_type, poly_type, wcoeffs_ortho, x, M, embedded_superdegree,
      interpolation_nderivs);
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
