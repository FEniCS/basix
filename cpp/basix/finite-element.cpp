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
#include "math.h"
#include "polyset.h"
#include "version.h"
#include <numeric>
#include <xtensor/xbuilder.hpp>

#define str_macro(X) #X
#define str(X) str_macro(X)

using namespace basix;

namespace
{
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
xt::xtensor<double, 2>
compute_dual_matrix(cell::type cell_type, const xt::xtensor<double, 2>& B,
                    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
                    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
                    int degree)
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

  std::size_t pdim = polyset::dim(cell_type, degree);
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
      if (x_e.shape(1) == 1 and x_e.shape(0) != 0)
      {
        auto pts = xt::view(x_e, xt::all(), 0);
        P = xt::view(polyset::tabulate(cell_type, degree, 0, pts), 0, xt::all(),
                     xt::all());
      }
      else if (x_e.shape(0) != 0)
      {
        P = xt::view(polyset::tabulate(cell_type, degree, 0, x_e), 0, xt::all(),
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

      dof_index += M[d][e].shape(0);
    }
  }

  /// Flatten D and take transpose
  auto Dt_flat = xt::transpose(
      xt::reshape_view(D, {D.shape(0), D.shape(1) * D.shape(2)}));

  return math::dot(B, Dt_flat);
}
//-----------------------------------------------------------------------------
} // namespace
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree,
                                           bool discontinuous)
{
  switch (family)
  {
  case element::family::P:
    if (degree < 3)
    {
      return element::create_lagrange(
          cell, degree, element::lagrange_variant::equispaced, discontinuous);
    }
    else
    {
      throw std::runtime_error(
          "Lagrange elements of degree > 2 need to be given a variant.");
    }
  case element::family::BDM:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_serendipity_div(cell, degree, discontinuous);
    case cell::type::hexahedron:
      return element::create_serendipity_div(cell, degree, discontinuous);
    default:
      return element::create_bdm(cell, degree, discontinuous);
    }
  case element::family::RT:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_rtc(cell, degree, discontinuous);
    case cell::type::hexahedron:
      return element::create_rtc(cell, degree, discontinuous);
    default:
      return element::create_rt(cell, degree, discontinuous);
    }
  }
  case element::family::N1E:
  {
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_nce(cell, degree, discontinuous);
    case cell::type::hexahedron:
      return element::create_nce(cell, degree, discontinuous);
    default:
      return element::create_nedelec(cell, degree, discontinuous);
    }
  }
  case element::family::N2E:
    switch (cell)
    {
    case cell::type::quadrilateral:
      return element::create_serendipity_curl(cell, degree, discontinuous);
    case cell::type::hexahedron:
      return element::create_serendipity_curl(cell, degree, discontinuous);
    default:
      return element::create_nedelec2(cell, degree, discontinuous);
    }
  case element::family::Regge:
    return element::create_regge(cell, degree, discontinuous);
  case element::family::CR:
    return element::create_cr(cell, degree, discontinuous);
  case element::family::bubble:
    return element::create_bubble(cell, degree, discontinuous);
  case element::family::serendipity:
    return element::create_serendipity(cell, degree, discontinuous);
  case element::family::DPC:
    return element::create_dpc(cell, degree, discontinuous);
  default:
    throw std::runtime_error("Element family not found");
  }
}
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree,
                                           element::lagrange_variant variant,
                                           bool discontinuous)
{
  switch (family)
  {
  case element::family::P:
    return element::create_lagrange(cell, degree, variant, discontinuous);
  default:
    throw std::runtime_error("Cannot pass a Lagrange variant.");
  }
}
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree,
                                           element::lagrange_variant variant)
{
  return create_element(family, cell, degree, variant, false);
}
//-----------------------------------------------------------------------------
basix::FiniteElement basix::create_element(element::family family,
                                           cell::type cell, int degree)
{
  return create_element(family, cell, degree, false);
}
//-----------------------------------------------------------------------------
std::tuple<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
           std::array<std::vector<xt::xtensor<double, 3>>, 4>,
           std::map<cell::type, xt::xtensor<double, 3>>>
basix::element::make_discontinuous(
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    std::map<cell::type, xt::xtensor<double, 3>>& entity_transformations,
    int tdim, int value_size)
{
  std::size_t npoints = 0;
  std::size_t Mshape0 = 0;
  for (int i = 0; i < 4; ++i)
  {
    for (std::size_t j = 0; j < x[i].size(); ++j)
    {
      npoints += x[i][j].shape(0);
      Mshape0 += M[i][j].shape(0);
    }
  }

  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations_out;
  std::array<std::vector<xt::xtensor<double, 3>>, 4> M_out;
  std::array<std::vector<xt::xtensor<double, 2>>, 4> x_out;

  xt::xtensor<double, 2> new_x
      = xt::zeros<double>({npoints, static_cast<std::size_t>(tdim)});

  xt::xtensor<double, 3> new_M = xt::zeros<double>(
      {Mshape0, static_cast<std::size_t>(value_size), npoints});

  int x_n = 0;
  int M_n = 0;
  for (int i = 0; i < 4; ++i)
  {
    for (std::size_t j = 0; j < x[i].size(); ++j)
    {
      xt::view(new_x, xt::range(x_n, x_n + x[i][j].shape(0)), xt::all())
          .assign(x[i][j]);
      xt::view(new_M, xt::range(M_n, M_n + M[i][j].shape(0)), xt::all(),
               xt::range(x_n, x_n + x[i][j].shape(0)))
          .assign(M[i][j]);
      x_n += x[i][j].shape(0);
      M_n += M[i][j].shape(0);
    }
  }

  x_out[tdim].push_back(new_x);
  M_out[tdim].push_back(new_M);

  for (auto it = entity_transformations.begin();
       it != entity_transformations.end(); ++it)
  {
    entity_transformations_out[it->first]
        = xt::xtensor<double, 3>({it->second.shape(0), 0, 0});
  }

  return {x_out, M_out, entity_transformations_out};
}
//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(
    element::family family, cell::type cell_type, int degree,
    const std::vector<std::size_t>& value_shape,
    const xt::xtensor<double, 2>& wcoeffs,
    const std::map<cell::type, xt::xtensor<double, 3>>& entity_transformations,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    maps::type map_type, bool discontinuous, int highest_degree,
    int highest_complete_degree,
    std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
        tensor_factors,
    element::lagrange_variant lvariant)
    : _cell_type(cell_type), _cell_tdim(cell::topological_dimension(cell_type)),
      _cell_subentity_types(cell::subentity_types(cell_type)), _family(family),
      _lagrange_variant(lvariant), _degree(degree), _map_type(map_type),
      _entity_transformations(entity_transformations), _x(x),
      _discontinuous(discontinuous),
      _degree_bounds({highest_complete_degree, highest_degree}),
      _tensor_factors(tensor_factors)
{
  _dual_matrix = compute_dual_matrix(cell_type, wcoeffs, M, x, degree);
  xt::xtensor<double, 2> B_cmajor({wcoeffs.shape(0), wcoeffs.shape(1)});
  B_cmajor.assign(wcoeffs);
  // Compute C = (BD^T)^{-1} B
  auto result = math::solve(_dual_matrix, B_cmajor);

  _coeffs = xt::xtensor<double, 2>({result.shape(0), result.shape(1)});
  _coeffs.assign(result);

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
  const std::vector<std::vector<std::vector<std::vector<int>>>> connectivity
      = cell::sub_entity_connectivity(cell_type);
  _num_edofs.resize(_cell_tdim + 1);
  _edofs.resize(_cell_tdim + 1);
  int dof = 0;
  for (std::size_t d = 0; d < _num_edofs.size(); ++d)
  {
    _num_edofs[d].resize(cell::num_sub_entities(_cell_type, d), 0);
    _edofs[d].resize(cell::num_sub_entities(_cell_type, d));
    for (std::size_t e = 0; e < M[d].size(); ++e)
    {
      _num_edofs[d][e] = M[d][e].shape(0);
      for (int i = 0; i < _num_edofs[d][e]; ++i)
        _edofs[d][e].push_back(dof++);
    }
  }

  _num_e_closure_dofs.resize(_cell_tdim + 1);
  _e_closure_dofs.resize(_cell_tdim + 1);
  for (std::size_t d = 0; d < _num_edofs.size(); ++d)
  {
    _num_e_closure_dofs[d].resize(cell::num_sub_entities(_cell_type, d));
    _e_closure_dofs[d].resize(cell::num_sub_entities(_cell_type, d));
    for (std::size_t e = 0; e < _e_closure_dofs[d].size(); ++e)
    {
      for (std::size_t dim = 0; dim <= d; ++dim)
      {
        for (int c : connectivity[d][e][dim])
        {
          _num_e_closure_dofs[d][e] += _edofs[dim][c].size();
          for (int dof : _edofs[dim][c])
            _e_closure_dofs[d][e].push_back(dof);
        }
      }
      std::sort(_e_closure_dofs[d][e].begin(), _e_closure_dofs[d][e].end());
    }
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
  for (const auto& et : _entity_transformations)
  {
    for (std::size_t i = 0;
         _dof_transformations_are_permutations and i < et.second.shape(0); ++i)
    {
      for (std::size_t row = 0; row < et.second.shape(1); ++row)
      {
        double rmin = xt::amin(xt::view(et.second, i, row, xt::all()))(0);
        double rmax = xt::amax(xt::view(et.second, i, row, xt::all()))(0);
        double rtot = xt::sum(xt::view(et.second, i, row, xt::all()))(0);
        if ((et.second.shape(2) != 1 and !xt::allclose(rmin, 0))
            or !xt::allclose(rmax, 1) or !xt::allclose(rtot, 1))
        {
          _dof_transformations_are_permutations = false;
          _dof_transformations_are_identity = false;
          break;
        }
        if (!xt::allclose(et.second(i, row, row), 1))
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
      for (const auto& et : _entity_transformations)
      {
        _eperm[et.first]
            = std::vector<std::vector<std::size_t>>(et.second.shape(0));
        _eperm_rev[et.first]
            = std::vector<std::vector<std::size_t>>(et.second.shape(0));
        for (std::size_t i = 0; i < et.second.shape(0); ++i)
        {
          std::vector<std::size_t> perm(et.second.shape(1));
          std::vector<std::size_t> rev_perm(et.second.shape(1));
          for (std::size_t row = 0; row < et.second.shape(1); ++row)
          {
            for (std::size_t col = 0; col < et.second.shape(1); ++col)
            {
              if (et.second(i, row, col) > 0.5)
              {
                perm[row] = col;
                rev_perm[col] = row;
                break;
              }
            }
          }
          // Factorise the permutations
          _eperm[et.first][i] = precompute::prepare_permutation(perm);
          _eperm_rev[et.first][i] = precompute::prepare_permutation(rev_perm);
        }
      }
    }

    // Precompute the DOF transformations
    for (const auto& et : _entity_transformations)
    {
      _etrans[et.first] = std::vector<
          std::tuple<std::vector<std::size_t>, std::vector<double>,
                     xt::xtensor<double, 2>>>(et.second.shape(0));
      _etransT[et.first] = std::vector<
          std::tuple<std::vector<std::size_t>, std::vector<double>,
                     xt::xtensor<double, 2>>>(et.second.shape(0));
      _etrans_invT[et.first] = std::vector<
          std::tuple<std::vector<std::size_t>, std::vector<double>,
                     xt::xtensor<double, 2>>>(et.second.shape(0));
      _etrans_inv[et.first] = std::vector<
          std::tuple<std::vector<std::size_t>, std::vector<double>,
                     xt::xtensor<double, 2>>>(et.second.shape(0));
      for (std::size_t i = 0; i < et.second.shape(0); ++i)
      {
        if (et.second.shape(1) > 0)
        {
          const xt::xtensor<double, 2>& M
              = xt::view(et.second, i, xt::all(), xt::all());
          _etrans[et.first][i] = precompute::prepare_matrix(M);
          auto M_t = xt::transpose(M);
          _etransT[et.first][i] = precompute::prepare_matrix(M_t);

          xt::xtensor<double, 2> Minv;
          // Rotation of a face: this is in the only base transformation such
          // that M^{-1} != M.
          // For a quadrilateral face, M^4 = Id, so M^{-1} = M^3.
          // For a triangular face, M^3 = Id, so M^{-1} = M^2.
          // This assumes that all faces of the cell are the same shape. For
          // prisms and pyramids, this will need updating to look at the face
          // type
          if (et.first == cell::type::quadrilateral and i == 0)
            Minv = math::dot(math::dot(M, M), M);
          else if (et.first == cell::type::triangle and i == 0)
            Minv = math::dot(M, M);
          else
            Minv = M;

          _etrans_inv[et.first][i] = precompute::prepare_matrix(Minv);
          auto MinvT = xt::transpose(Minv);
          _etrans_invT[et.first][i] = precompute::prepare_matrix(MinvT);
        }
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
maps::type FiniteElement::map_type() const { return _map_type; }
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
const xt::xtensor<double, 2>& FiniteElement::interpolation_matrix() const
{
  return _matM;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<int>>& FiniteElement::num_entity_dofs() const
{
  return _num_edofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
FiniteElement::entity_dofs() const
{
  return _edofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<int>>&
FiniteElement::num_entity_closure_dofs() const
{
  return _num_e_closure_dofs;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<std::vector<int>>>&
FiniteElement::entity_closure_dofs() const
{
  return _e_closure_dofs;
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
  std::size_t vs = value_size();
  std::size_t ndofs = _coeffs.shape(0);
  return {ndsize, num_points, ndofs, vs};
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 4>
FiniteElement::tabulate(int nd, const xt::xarray<double>& x) const
{
  xt::xarray<double> _x = x;
  if (_x.dimension() == 1)
    _x.reshape({_x.shape(0), 1});

  auto shape = tabulate_shape(nd, x.shape(0));
  xt::xtensor<double, 4> data(shape);
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

  xt::xtensor<double, 3> basis(
      {static_cast<std::size_t>(polyset::nderivs(_cell_type, nd)), _x.shape(0),
       static_cast<std::size_t>(polyset::dim(_cell_type, _degree))});
  polyset::tabulate(basis, _cell_type, _degree, nd, _x);
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
      auto result = math::dot(B, xt::transpose(C));
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
  {
    dof_start
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);
  }

  if (_cell_tdim > 1)
  {
    // Base transformations for edges

    for (int ndofs : _num_edofs[1])
    {
      xt::view(bt, transform_n++, xt::range(dof_start, dof_start + ndofs),
               xt::range(dof_start, dof_start + ndofs))
          = xt::view(_entity_transformations.at(cell::type::interval), 0,
                     xt::all(), xt::all());
      dof_start += ndofs;
    }

    if (_cell_tdim > 2)
    {
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        const int ndofs = _num_edofs[2][f];
        if (ndofs > 0)
        {
          // TODO: This assumes that every face has the same shape
          //       _entity_transformations should be replaced with a map from
          //       a subentity type to a matrix to allow for prisms and
          //       pyramids.
          xt::view(bt, transform_n++, xt::range(dof_start, dof_start + ndofs),
                   xt::range(dof_start, dof_start + ndofs))
              = xt::view(
                  _entity_transformations.at(_cell_subentity_types[2][f]), 0,
                  xt::all(), xt::all());
          xt::view(bt, transform_n++, xt::range(dof_start, dof_start + ndofs),
                   xt::range(dof_start, dof_start + ndofs))
              = xt::view(
                  _entity_transformations.at(_cell_subentity_types[2][f]), 1,
                  xt::all(), xt::all());

          dof_start += ndofs;
        }
      }
    }
  }

  return bt;
}
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& FiniteElement::points() const { return _points; }
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> FiniteElement::push_forward(
    const xt::xtensor<double, 3>& U, const xt::xtensor<double, 3>& J,
    const xtl::span<const double>& detJ, const xt::xtensor<double, 3>& K) const
{
  const std::size_t physical_value_size
      = compute_value_size(_map_type, J.shape(1));
  xt::xtensor<double, 3> u({U.shape(0), U.shape(1), physical_value_size});
  using u_t = xt::xview<decltype(u)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(U)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto map = this->map_fn<u_t, U_t, J_t, K_t>();
  for (std::size_t i = 0; i < u.shape(0); ++i)
  {
    auto _K = xt::view(K, i, xt::all(), xt::all());
    auto _J = xt::view(J, i, xt::all(), xt::all());
    auto _u = xt::view(u, i, xt::all(), xt::all());
    auto _U = xt::view(U, i, xt::all(), xt::all());
    map(_u, _U, _J, detJ[i], _K);
  }

  return u;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> FiniteElement::pull_back(
    const xt::xtensor<double, 3>& u, const xt::xtensor<double, 3>& J,
    const xtl::span<const double>& detJ, const xt::xtensor<double, 3>& K) const
{
  const std::size_t reference_value_size = value_size();
  xt::xtensor<double, 3> U({u.shape(0), u.shape(1), reference_value_size});
  using u_t = xt::xview<decltype(u)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(U)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto map = this->map_fn<U_t, u_t, K_t, J_t>();
  for (std::size_t i = 0; i < u.shape(0); ++i)
  {
    auto _K = xt::view(K, i, xt::all(), xt::all());
    auto _J = xt::view(J, i, xt::all(), xt::all());
    auto _u = xt::view(u, i, xt::all(), xt::all());
    auto _U = xt::view(U, i, xt::all(), xt::all());
    map(_U, _u, _K, 1.0 / detJ[i], _J);
  }

  return U;
}
//-----------------------------------------------------------------------------
void FiniteElement::permute_dofs(const xtl::span<std::int32_t>& dofs,
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
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Permute DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        precompute::apply_permutation(_eperm.at(cell::type::interval)[0], dofs,
                                      dofstart);
      }
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
        {
          precompute::apply_permutation(
              _eperm.at(_cell_subentity_types[2][f])[1], dofs, dofstart);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          precompute::apply_permutation(
              _eperm.at(_cell_subentity_types[2][f])[0], dofs, dofstart);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::unpermute_dofs(const xtl::span<std::int32_t>& dofs,
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
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Permute DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        precompute::apply_permutation(_eperm_rev.at(cell::type::interval)[0],
                                      dofs, dofstart);
      }
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          precompute::apply_permutation(
              _eperm_rev.at(_cell_subentity_types[2][f])[0], dofs, dofstart);
        }

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
        {
          precompute::apply_permutation(
              _eperm_rev.at(_cell_subentity_types[2][f])[1], dofs, dofstart);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::map<cell::type, xt::xtensor<double, 3>>
FiniteElement::entity_transformations() const
{
  return _entity_transformations;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> FiniteElement::dual_matrix() const
{
  return _dual_matrix;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2> FiniteElement::coefficient_matrix() const
{
  return _coeffs;
}
//-----------------------------------------------------------------------------
std::array<int, 2> FiniteElement::degree_bounds() const
{
  return _degree_bounds;
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
std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
FiniteElement::get_tensor_product_representation() const
{
  if (!has_tensor_product_factorisation())
    throw std::runtime_error("Element has no tensor product representation.");
  return _tensor_factors;
}
//-----------------------------------------------------------------------------
bool basix::elements_are_equal(FiniteElement element1, FiniteElement element2)
{
  return element1.cell_type() == element2.cell_type()
         and element1.family() == element2.family()
         and element1.degree() == element2.degree()
         and element1.discontinuous() == element2.discontinuous()
         and element1.lagrange_variant() == element2.lagrange_variant();
}
//-----------------------------------------------------------------------------
std::string basix::version()
{
  static const std::string version_str = str(BASIX_VERSION);
  return version_str;
}
//-----------------------------------------------------------------------------
