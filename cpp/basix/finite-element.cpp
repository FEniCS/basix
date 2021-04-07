// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "brezzi-douglas-marini.h"
#include "bubble.h"
#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nce-rtc.h"
#include "nedelec.h"
#include "polyset.h"
#include "raviart-thomas.h"
#include "regge.h"
#include "serendipity.h"
#include <numeric>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor/xview.hpp>

#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;

namespace
{
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
} // namespace
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(std::string family, std::string cell,
                                           int degree)
{
  return basix::create_element(element::str_to_type(family),
                               cell::str_to_type(cell), degree);
}
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree)
{
  switch (family)
  {
  case element::family::P:
    return create_lagrange(cell, degree);
  case element::family::DP:
    return create_dlagrange(cell, degree);
  case element::family::BDM:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_serendipity_div(cell, degree);
    case cell::type::hexahedron:
      return create_serendipity_div(cell, degree);
    default:
      return create_bdm(cell, degree);
    }
  case element::family::RT:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_rtc(cell, degree);
    case cell::type::hexahedron:
      return create_rtc(cell, degree);
    default:
      return create_rt(cell, degree);
    }
  }
  case element::family::N1E:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_nce(cell, degree);
    case cell::type::hexahedron:
      return create_nce(cell, degree);
    default:
      return create_nedelec(cell, degree);
    }
  }
  case element::family::N2E:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_serendipity_curl(cell, degree);
    case cell::type::hexahedron:
      return create_serendipity_curl(cell, degree);
    default:
      return create_nedelec2(cell, degree);
    }
  case element::family::Regge:
    return create_regge(cell, degree);
  case element::family::CR:
    return create_cr(cell, degree);
  case element::family::Bubble:
    return create_bubble(cell, degree);
  case element::family::Serendipity:
    return create_serendipity(cell, degree);
  case element::family::DPC:
    return create_dpc(cell, degree);
  default:
    throw std::runtime_error("Element family not found");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> basix::compute_expansion_coefficients(
    cell::type celltype, const xt::xtensor<double, 2>& B,
    const std::vector<std::vector<xt::xtensor<double, 3>>>& M,
    const std::vector<std::vector<xt::xtensor<double, 2>>>& x, int degree,
    double kappa_tol)
{
  std::size_t num_dofs(0), vs(0);
  for (auto& Md : M)
  {
    for (auto& Me : Md)
    {
      num_dofs += Me.shape(0);
      if (vs == 0)
        vs = Me.shape(1);
      else if (vs != Me.shape(1))
        throw std::runtime_error("Inconsistent value size");
    }
  }

  std::size_t pdim = polyset::dim(celltype, degree);
  xt::xtensor<double, 3> D = xt::zeros<double>({num_dofs, vs, pdim});

  // Loop over different dimensions
  std::size_t dof_index = 0;
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    // Loop over entities of dimension d
    for (std::size_t e = 0; e < x[d].size(); ++e)
    {
      // Evaluate polynomial basis at x[d]
      const xt::xtensor<double, 2>& x_e = x[d][e];
      xt::xtensor<double, 2> P;
      if (x_e.shape(1) == 1 and x_e.size() != 0)
      {
        auto pts = xt::view(x_e, xt::all(), 0);
        P = xt::view(polyset::tabulate(celltype, degree, 0, pts), 0, xt::all(),
                     xt::all());
      }
      else if (x_e.size() != 0)
      {
        P = xt::view(polyset::tabulate(celltype, degree, 0, x_e), 0, xt::all(),
                     xt::all());
      }

      // Me: [dof, vs, point]
      const xt::xtensor<double, 3>& Me = M[d][e];

      // Compute dual matrix contribution
      for (std::size_t i = 0; i < Me.shape(0); ++i)      // Dof index
        for (std::size_t j = 0; j < Me.shape(1); ++j)    // Value index
          for (std::size_t k = 0; k < Me.shape(2); ++k)  // Point
            for (std::size_t l = 0; l < P.shape(1); ++l) // Polynomial term
              D(dof_index + i, j, l) += Me(i, j, k) * P(k, l);

      // Dtmp += xt::linalg::dot(Me, P);

      dof_index += M[d][e].shape(0);
    }
  }

  // Compute B D^{T}
  // xt::xtensor<double, 2> A = xt::zeros<double>({num_dofs, num_dofs});
  // for (std::size_t i = 0; i < A.shape(0); ++i)
  //   for (std::size_t j = 0; j < A.shape(1); ++j)
  //     for (std::size_t k = 0; k < vs; ++k)
  //       for (std::size_t l = 0; l < B[k].shape(1); ++l)
  //         A(i, j) += B[k](i, l) * D(j, k, l);

  /// Flatten D and take transpose
  auto Dt_flat = xt::transpose(
      xt::reshape_view(D, {D.shape(0), D.shape(1) * D.shape(2)}));

  xt::xtensor<double, 2, xt::layout_type::column_major> BDt
      = xt::linalg::dot(B, Dt_flat);

  if (kappa_tol >= 1.0)
  {
    if (xt::linalg::cond(BDt, 2) > kappa_tol)
    {
      throw std::runtime_error("Condition number of B.D^T when computing "
                               "expansion coefficients exceeds tolerance.");
    }
  }

  // FIXME: Workaround for an issue in xtensor and intel compilers
  xt::xtensor<double, 2, xt::layout_type::column_major> B_cmajor(
      {B.shape(0), B.shape(1)});
  B_cmajor.assign(B);

  // Compute C = (BD^T)^{-1} B
  auto result = xt::linalg::solve(BDt, B_cmajor);

  xt::xtensor<double, 2> C({result.shape(0), result.shape(1)});
  C.assign(result);

  return xt::reshape_view(C, {num_dofs, vs, pdim});
}
//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(
    element::family family, cell::type cell_type, int degree,
    const std::vector<std::size_t>& value_shape,
    const xt::xtensor<double, 3>& coeffs,
    const xt::xtensor<double, 3>& base_transformations,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    maps::type map_type)
    : map_type(map_type), _cell_type(cell_type), _family(family),
      _degree(degree), _map_type(map_type),
      _coeffs(xt::reshape_view(
          coeffs, {coeffs.shape(0), coeffs.shape(1) * coeffs.shape(2)})),
      _base_transformations(base_transformations), _x(x), _matM_new(M)
{
  // if (points.dimension() == 1)
  //   throw std::runtime_error("Problem with points");

  _value_shape = std::vector<int>(value_shape.begin(), value_shape.end());

  std::size_t num_points = 0;
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      num_points += x_e.shape(0);

  std::size_t tdim = geometry(cell_type).shape(1);
  std::size_t counter = 0;
  _points.resize({num_points, tdim});
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      for (std::size_t p = 0; p < x_e.shape(0); ++p)
        xt::row(_points, counter++) = xt::row(x_e, p);

  // Copy into _matM

  const std::size_t value_size
      = std::accumulate(value_shape.begin(), value_shape.end(), 1,
                        std::multiplies<std::size_t>());

  // Count number of dofs and point
  std::size_t num_dofs(0), num_points1(0);
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    for (std::size_t e = 0; e < M[d].size(); ++e)
    {
      num_dofs += M[d][e].shape(0);
      num_points1 += M[d][e].shape(2);
    }
  }

  // Copy data into old _matM matrix

  _matM = xt::zeros<double>({num_dofs, value_size * num_points1});
  auto Mview = xt::reshape_view(_matM, {num_dofs, value_size, num_points1});

  // Loop over each topological dimensions
  std::size_t dof_offset(0), point_offset(0);
  for (std::size_t d = 0; d < M.size(); ++d)
  {
    // Loop of entities of dimension d
    for (std::size_t e = 0; e < M[d].size(); ++e)
    {
      auto dof_range = xt::range(dof_offset, dof_offset + M[d][e].shape(0));
      auto point_range
          = xt::range(point_offset, point_offset + M[d][e].shape(2));
      xt::view(Mview, dof_range, xt::all(), point_range) = M[d][e];
      point_offset += M[d][e].shape(2);
      dof_offset += M[d][e].shape(0);
    }
  }

  // Compute number of dofs for each cell entity (computed from
  // interpolation data)
  const std::vector<std::vector<std::vector<int>>> topology
      = cell::topology(cell_type);
  _entity_dofs.resize(tdim + 1);
  for (std::size_t d = 0; d < _entity_dofs.size(); ++d)
  {
    _entity_dofs[d].resize(topology[d].size(), 0);
    for (std::size_t e = 0; e < M[d].size(); ++e)
      _entity_dofs[d][e] = M[d][e].shape(0);
  }

  // Check that nunber of dofs os equal to number of coefficients
  if (num_dofs != _coeffs.shape(0))
  {
    throw std::runtime_error(
        "Number of entity dofs does not match total number of dofs");
  }
}
//-----------------------------------------------------------------------------
cell::type FiniteElement::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
int FiniteElement::degree() const { return _degree; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const
{
  return std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                         std::multiplies<int>());
}
//-----------------------------------------------------------------------------
const std::vector<int>& FiniteElement::value_shape() const
{
  return _value_shape;
}
//-----------------------------------------------------------------------------
int FiniteElement::dim() const { return _coeffs.shape(0); }
//-----------------------------------------------------------------------------
element::family FiniteElement::family() const { return _family; }
//-----------------------------------------------------------------------------
maps::type FiniteElement::mapping_type() const { return _map_type; }
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& FiniteElement::interpolation_matrix() const
{
  return _matM;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<int>>& FiniteElement::entity_dofs() const
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 4>
FiniteElement::tabulate(int nd, const xt::xarray<double>& x) const
{
  const std::size_t tdim = cell::topological_dimension(_cell_type);
  std::size_t ndsize = 1;
  for (int i = 1; i <= nd; ++i)
    ndsize *= (tdim + i);
  for (int i = 1; i <= nd; ++i)
    ndsize /= i;
  const std::size_t vs = value_size();
  const std::size_t ndofs = _coeffs.shape(0);

  xt::xarray<double> _x = x;
  if (_x.dimension() == 1)
    _x.reshape({_x.shape(0), 1});

  // Tabulate
  std::vector<double> basis_data(ndsize * x.shape(0) * ndofs * vs);
  tabulate(nd, _x, basis_data.data());

  // Pack data in
  xt::xtensor<double, 4> data({ndsize, _x.shape(0), ndofs, vs});

  // Loop over derivatives
  for (std::size_t d = 0; d < data.shape(0); ++d)
  {
    std::size_t offset_d = d * data.shape(1) * data.shape(2) * data.shape(3);

    // Loop over points
    for (std::size_t p = 0; p < data.shape(1); ++p)
    {
      // Loop over basis functions
      for (std::size_t r = 0; r < data.shape(2); ++r)
      {
        // Loop over values
        for (std::size_t i = 0; i < data.shape(3); ++i)
        {
          std::size_t offset = offset_d + p + r * data.shape(1)
                               + i * data.shape(1) * data.shape(2);
          assert(offset < basis_data.size());
          data(d, p, r, i) = basis_data[offset];
        }
      }
    }
  }

  return data;
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate(int nd, const xt::xarray<double>& x,
                             double* basis_data) const
{
  xt::xarray<double> _x = x;
  if (_x.dimension() == 2 and x.shape(1) == 1)
    _x.reshape({x.shape(0)});

  const std::size_t tdim = cell::topological_dimension(_cell_type);
  if (_x.shape(1) != tdim)
    throw std::runtime_error("Point dim does not match element dim.");

  xt::xtensor<double, 3> basis = polyset::tabulate(_cell_type, _degree, nd, _x);
  const int psize = polyset::dim(_cell_type, _degree);
  const std::size_t ndofs = _coeffs.shape(0);
  const int vs = value_size();
  xt::xtensor<double, 2> B, C;
  for (std::size_t p = 0; p < basis.shape(0); ++p)
  {
    // Map block for current derivative
    std::array<std::size_t, 2> shape = {_x.shape(0), ndofs * vs};
    std::size_t offset = p * x.shape(0) * ndofs * vs;
    auto dresult = xt::adapt<xt::layout_type::column_major>(
        basis_data + offset, x.shape(0) * ndofs * vs, xt::no_ownership(),
        shape);
    for (int j = 0; j < vs; ++j)
    {
      B = xt::view(basis, p, xt::all(), xt::all());
      C = xt::transpose(xt::view(_coeffs, xt::all(),
                                 xt::range(psize * j, psize * j + psize)));
      xt::view(dresult, xt::range(0, x.shape(0)),
               xt::range(ndofs * j, ndofs * j + ndofs))
          = xt::linalg::dot(B, C);
    }
  }
}
//-----------------------------------------------------------------------------
const xt::xtensor<double, 3>& FiniteElement::base_transformations() const
{
  return _base_transformations;
}
//-----------------------------------------------------------------------------
int FiniteElement::num_points() const { return _points.shape(0); }
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& FiniteElement::points() const { return _points; }
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> FiniteElement::map_push_forward(
    const xt::xtensor<double, 3>& U, const xt::xtensor<double, 3>& J,
    const tcb::span<const double>& detJ, const xt::xtensor<double, 3>& K) const
{
  const std::size_t physical_value_size
      = compute_value_size(_map_type, J.shape(1));
  xt::xtensor<double, 3> u({U.shape(0), U.shape(1), physical_value_size});
  map_push_forward_m(U, J, detJ, K, u);

  return u;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> FiniteElement::map_pull_back(
    const xt::xtensor<double, 3>& u, const xt::xtensor<double, 3>& J,
    const tcb::span<const double>& detJ, const xt::xtensor<double, 3>& K) const
{
  const std::size_t reference_value_size = value_size();
  xt::xtensor<double, 3> U({u.shape(0), u.shape(1), reference_value_size});
  map_pull_back_m(u, J, detJ, K, U);
  return U;
}
//-----------------------------------------------------------------------------
std::string basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
