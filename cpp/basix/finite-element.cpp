// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "e-brezzi-douglas-marini.h"
#include "e-bubble.h"
#include "e-crouzeix-raviart.h"
#include "e-lagrange.h"
#include "e-nce-rtc.h"
#include "e-nedelec.h"
#include "e-raviart-thomas.h"
#include "e-regge.h"
#include "e-serendipity.h"
#include "polyset.h"
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
} // namespace
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(std::string family, std::string cell,
                                           int degree, std::string variant)
{
  return basix::create_element(element::str_to_family(family),
                               cell::str_to_type(cell), degree,
                               element::str_to_variant(variant));
}
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree,
                                           element::variant variant)
{
  switch (family)
  {
  case element::family::P:
    return create_lagrange(cell, degree, variant);
  case element::family::DP:
    return create_dlagrange(cell, degree, variant);
  case element::family::BDM:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_serendipity_div(cell, degree, variant);
    case cell::type::hexahedron:
      return create_serendipity_div(cell, degree, variant);
    default:
      return create_bdm(cell, degree, variant);
    }
  case element::family::RT:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_rtc(cell, degree, variant);
    case cell::type::hexahedron:
      return create_rtc(cell, degree, variant);
    default:
      return create_rt(cell, degree, variant);
    }
  }
  case element::family::N1E:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_nce(cell, degree, variant);
    case cell::type::hexahedron:
      return create_nce(cell, degree, variant);
    default:
      return create_nedelec(cell, degree, variant);
    }
  }
  case element::family::N2E:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return create_serendipity_curl(cell, degree, variant);
    case cell::type::hexahedron:
      return create_serendipity_curl(cell, degree, variant);
    default:
      return create_nedelec2(cell, degree, variant);
    }
  case element::family::Regge:
    return create_regge(cell, degree, variant);
  case element::family::CR:
    return create_cr(cell, degree, variant);
  case element::family::Bubble:
    return create_bubble(cell, degree, variant);
  case element::family::Serendipity:
    return create_serendipity(cell, degree, variant);
  case element::family::DPC:
    return create_dpc(cell, degree, variant);
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

  // Note: forcing the layout type to get around an xtensor bug with Intel
  // Compilers
  // https://github.com/xtensor-stack/xtensor/issues/2351
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
    const std::vector<xt::xtensor<double, 2>>& entity_transformations,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    maps::type map_type)
    : map_type(map_type), _cell_type(cell_type),
      _cell_tdim(cell::topological_dimension(cell_type)), _family(family),
      _degree(degree), _map_type(map_type),
      _coeffs(xt::reshape_view(
          coeffs, {coeffs.shape(0), coeffs.shape(1) * coeffs.shape(2)})),
      _entity_transformations(entity_transformations), _x(x), _matM_new(M)
{
  // if (points.dimension() == 1)
  //   throw std::runtime_error("Problem with points");

  _value_shape = std::vector<int>(value_shape.begin(), value_shape.end());

  std::size_t num_points = 0;
  for (auto& x_dim : x)
    for (auto& x_e : x_dim)
      num_points += x_e.shape(0);

  std::size_t counter = 0;
  _points.resize({num_points, _cell_tdim});
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
  _edofs.resize(_cell_tdim + 1);
  for (std::size_t d = 0; d < _edofs.size(); ++d)
  {
    _edofs[d].resize(cell::num_sub_entities(_cell_type, d), 0);
    for (std::size_t e = 0; e < M[d].size(); ++e)
      _edofs[d][e] = M[d][e].shape(0);
  }

  // Check that nunber of dofs os equal to number of coefficients
  if (num_dofs != _coeffs.shape(0))
  {
    throw std::runtime_error(
        "Number of entity dofs does not match total number of dofs");
  }

  // Check if base transformations are all permutations
  _dof_transformations_are_permutations = true;
  _dof_transformations_are_identity = true;
  for (std::size_t i = 0; _dof_transformations_are_permutations
                          and i < _entity_transformations.size();
       ++i)
  {
    for (std::size_t row = 0; row < _entity_transformations[i].shape(0); ++row)
    {
      double rmin
          = xt::amin(xt::view(_entity_transformations[i], row, xt::all()))(0);
      double rmax
          = xt::amax(xt::view(_entity_transformations[i], row, xt::all()))(0);
      double rtot
          = xt::sum(xt::view(_entity_transformations[i], row, xt::all()))(0);
      if ((_entity_transformations[i].shape(1) != 1 and !xt::allclose(rmin, 0))
          or !xt::allclose(rmax, 1) or !xt::allclose(rtot, 1))
      {
        _dof_transformations_are_permutations = false;
        _dof_transformations_are_identity = false;
        break;
      }
      if (!xt::allclose(_entity_transformations[i](row, row), 1))
        _dof_transformations_are_identity = false;
    }
  }
  if (!_dof_transformations_are_identity)
  {
    // If transformations are permutations, then create the permutations
    if (_dof_transformations_are_permutations)
    {
      for (std::size_t i = 0; i < _entity_transformations.size(); ++i)
      {
        std::vector<std::size_t> perm(_entity_transformations[i].shape(0));
        std::vector<std::size_t> rev_perm(_entity_transformations[i].shape(0));
        for (std::size_t row = 0; row < _entity_transformations[i].shape(0);
             ++row)
        {
          for (std::size_t col = 0; col < _entity_transformations[i].shape(0);
               ++col)
          {
            if (_entity_transformations[i](row, col) > 0.5)
            {
              perm[row] = col;
              rev_perm[col] = row;
              break;
            }
          }
        }
        // Factorise the permutations
        _eperm.push_back(precompute::prepare_permutation(perm));
        _eperm_rev.push_back(precompute::prepare_permutation(rev_perm));
      }
    }

    // Precompute the DOF transformations
    _etrans.resize(_entity_transformations.size());
    _etrans_inv.resize(_entity_transformations.size());
    for (std::size_t i = 0; i < _entity_transformations.size(); ++i)
    {
      if (_entity_transformations[i].shape(0) > 0)
      {
        const xt::xtensor<double, 2>& M = _entity_transformations[i];
        _etrans[i] = precompute::prepare_matrix(M);

        xt::xtensor<double, 2> Minv;
        if (i == 1)
        {
          // Rotation of a face: this is in the only base transformation such
          // that M^{-1} != M.
          // For a quadrilateral face, M^4 = Id, so M^{-1} = M^3.
          // For a triangular face, M^3 = Id, so M^{-1} = M^2.
          // This assumes that all faces of the cell are the same shape. For
          // prisms and pyramids, this will need updating to look at the face
          // type
          if (_cell_type == cell::type::hexahedron)
            Minv = xt::linalg::dot(xt::linalg::dot(M, M), M);
          else
            Minv = xt::linalg::dot(M, M);
        }
        else
          Minv = M;

        auto MinvT = xt::transpose(Minv);
        _etrans_inv[i] = precompute::prepare_matrix(MinvT);
      }
    }
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
const xt::xtensor<double, 2>& FiniteElement::interpolation_matrix() const
{
  return _matM;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<int>>& FiniteElement::entity_dofs() const
{
  return _edofs;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 4>
FiniteElement::tabulate(int nd, const xt::xarray<double>& x) const
{
  std::size_t ndsize = 1;
  for (int i = 1; i <= nd; ++i)
    ndsize *= (_cell_tdim + i);
  for (int i = 1; i <= nd; ++i)
    ndsize /= i;
  const std::size_t vs = value_size();
  const std::size_t ndofs = _coeffs.shape(0);

  xt::xarray<double> _x = x;
  if (_x.dimension() == 1)
    _x.reshape({_x.shape(0), 1});

  xt::xtensor<double, 4> data({ndsize, x.shape(0), ndofs, vs});
  tabulate(nd, _x, data);

  return data;
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate(int nd, const xt::xarray<double>& x,
                             xt::xtensor<double, 4>& basis_data) const
{
  xt::xarray<double> _x = x;
  if (_x.dimension() == 2 and x.shape(1) == 1)
    _x.reshape({x.shape(0)});

  if (_x.shape(1) != _cell_tdim)
    throw std::runtime_error("Point dim does not match element dim.");

  xt::xtensor<double, 3> basis = polyset::tabulate(_cell_type, _degree, nd, _x);
  const int psize = polyset::dim(_cell_type, _degree);
  const int vs = value_size();
  xt::xtensor<double, 2> B, C;
  for (std::size_t p = 0; p < basis.shape(0); ++p)
  {
    for (int j = 0; j < vs; ++j)
    {
      auto basis_view = xt::view(basis_data, p, xt::all(), xt::all(), j);
      B = xt::view(basis, p, xt::all(), xt::all());
      C = xt::view(_coeffs, xt::all(), xt::range(psize * j, psize * j + psize));
      auto result = xt::linalg::dot(B, xt::transpose(C));
      basis_view.assign(result);
    }
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> FiniteElement::base_transformations() const
{
  const std::size_t nt = num_transformations(cell_type());
  const std::size_t ndofs = this->dim();

  xt::xtensor<double, 3> bt({nt, ndofs, ndofs});
  for (std::size_t i = 0; i < nt; ++i)
    xt::view(bt, i, xt::all(), xt::all()) = xt::eye<double>(ndofs);

  std::size_t dof_start = 0;
  int transform_n = 0;
  if (_cell_tdim > 0)
    dof_start = std::accumulate(_edofs[0].cbegin(), _edofs[0].cend(), 0);

  if (_cell_tdim > 1)
  {
    // Base transformations for edges

    for (int ndofs : _edofs[1])
    {
      xt::view(bt, transform_n++, xt::range(dof_start, dof_start + ndofs),
               xt::range(dof_start, dof_start + ndofs))
          = _entity_transformations[0];
      dof_start += ndofs;
    }

    if (_cell_tdim > 2)
    {
      for (int ndofs : _edofs[2])
      {
        if (ndofs > 0)
        {
          // TODO: This assumes that every face has the same shape
          //       _entity_transformations should be replaced with a map from a
          //       subentity type to a matrix to allow for prisms and pyramids.
          xt::view(bt, transform_n++, xt::range(dof_start, dof_start + ndofs),
                   xt::range(dof_start, dof_start + ndofs))
              = _entity_transformations[1];
          xt::view(bt, transform_n++, xt::range(dof_start, dof_start + ndofs),
                   xt::range(dof_start, dof_start + ndofs))
              = _entity_transformations[2];

          dof_start += ndofs;
        }
      }
    }
  }

  return bt;
}
//-----------------------------------------------------------------------------
int FiniteElement::num_points() const { return _points.shape(0); }
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& FiniteElement::points() const { return _points; }
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> FiniteElement::map_push_forward(
    const xt::xtensor<double, 3>& U, const xt::xtensor<double, 3>& J,
    const xtl::span<const double>& detJ, const xt::xtensor<double, 3>& K) const
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
    const xtl::span<const double>& detJ, const xt::xtensor<double, 3>& K) const
{
  const std::size_t reference_value_size = value_size();
  xt::xtensor<double, 3> U({u.shape(0), u.shape(1), reference_value_size});
  map_pull_back_m(u, J, detJ, K, U);
  return U;
}
//-----------------------------------------------------------------------------
void FiniteElement::permute_dofs(xtl::span<std::int32_t>& dofs,
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
    int dofstart = std::accumulate(_edofs[0].cbegin(), _edofs[0].cend(), 0);

    // Permute DOFs on edges
    for (std::size_t e = 0; e < _edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
        precompute::apply_permutation(_eperm[0], dofs, dofstart);
      dofstart += _edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_permutation(_eperm[2], dofs, dofstart);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_permutation(_eperm[1], dofs, dofstart);

        dofstart += _edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::unpermute_dofs(xtl::span<std::int32_t>& dofs,
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
    int dofstart = std::accumulate(_edofs[0].cbegin(), _edofs[0].cend(), 0);

    // Permute DOFs on edges
    for (std::size_t e = 0; e < _edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
        precompute::apply_permutation(_eperm_rev[0], dofs, dofstart);
      dofstart += _edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_permutation(_eperm_rev[1], dofs, dofstart);

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_permutation(_eperm_rev[2], dofs, dofstart);

        dofstart += _edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::string basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
