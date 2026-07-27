// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "e-nedelec.h"
#include "e-lagrange.h"
#include "e-raviart-thomas.h"
#include "element-families.h"
#include "maps.h"
#include "math.h"
#include "mdspan.hpp"
#include "moments.h"
#include "polyset.h"
#include "quadrature.h"
#include "sobolev-spaces.h"
#include <array>
#include <concepts>
#include <vector>

using namespace basix;

namespace
{
//-----------------------------------------------------------------------------
template <std::floating_point T>
impl::mdarray_t<T, 2> create_nedelec_2d_space(int degree)
{
  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) / 2;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree / 2;

  // Number of additional polynomials in Nedelec set
  const std::size_t ns = degree;

  // Tabulate polynomial set at quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, cell::type::triangle, polyset::type::standard,
      2 * degree);
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                 _pts.size() / wts.size());
  const auto [_phi, shape] = polyset::tabulate(
      cell::type::triangle, polyset::type::standard, degree, 0, pts);
  impl::mdspan_t<const T, 3> phi(_phi.data(), shape);

  const std::size_t psize = phi.extent(1);

  // Create coefficients for order (degree-1) vector polynomials
  impl::mdarray_t<T, 2> wcoeffs(nv * 2 + ns, psize * 2);
  for (std::size_t i = 0; i < nv; ++i)
  {
    wcoeffs(i, i) = 1.0;
    wcoeffs(nv + i, psize + i) = 1.0;
  }

  // Create coefficients for the additional Nedelec polynomials.
  //
  // For each (i, j) this originally computed a sum over quadrature points
  // k of wts[k]*phi(0,ns0+i,k)*pts(k,dim)*phi(0,j,k) (dim = 1 and 0) via
  // an explicit triple loop. Since phi(0,j,k) (for j in [nv,psize)) does
  // not depend on i, and wts[k]*pts(k,dim)*phi(0,ns0+i,k) does not depend
  // on j, this is a pair of matrix-matrix products, computed here via
  // BLAS (math::dot) instead.
  const std::size_t npts = wts.size();
  const std::size_t ncols = psize - nv;

  // B(k, jc) = phi(0, nv + jc, k)
  std::vector<T> Bb(npts * ncols);
  impl::mdspan_t<T, 2> B(Bb.data(), npts, ncols);
  for (std::size_t k = 0; k < npts; ++k)
    for (std::size_t jc = 0; jc < ncols; ++jc)
      B(k, jc) = phi(0, nv + jc, k);

  // A1(i, k) = wts[k] * pts(k, 1) * phi(0, ns0 + i, k)
  // A2(i, k) = wts[k] * pts(k, 0) * phi(0, ns0 + i, k)
  std::vector<T> A1b(ns * npts);
  std::vector<T> A2b(ns * npts);
  impl::mdspan_t<T, 2> A1(A1b.data(), ns, npts);
  impl::mdspan_t<T, 2> A2(A2b.data(), ns, npts);
  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t k = 0; k < npts; ++k)
    {
      const T p = wts[k] * phi(0, ns0 + i, k);
      A1(i, k) = p * pts(k, 1);
      A2(i, k) = p * pts(k, 0);
    }
  }

  std::vector<T> R1b(ns * ncols), R2b(ns * ncols);
  impl::mdspan_t<T, 2> R1(R1b.data(), ns, ncols);
  impl::mdspan_t<T, 2> R2(R2b.data(), ns, ncols);
  math::dot(A1, B, R1);
  math::dot(A2, B, R2);

  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t jc = 0; jc < ncols; ++jc)
    {
      wcoeffs(2 * nv + i, nv + jc) = R1(i, jc);
      wcoeffs(2 * nv + i, nv + jc + psize) = -R2(i, jc);
    }
  }

  math::orthogonalise<T>(wcoeffs, nv * 2);

  return wcoeffs;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
impl::mdarray_t<T, 2> create_nedelec_3d_space(int degree)
{
  // Reference tetrahedron
  const std::size_t tdim = 3;

  // Number of order (degree) vector polynomials
  const std::size_t nv = degree * (degree + 1) * (degree + 2) / 6;

  // Number of order (degree-1) vector polynomials
  const std::size_t ns0 = (degree - 1) * degree * (degree + 1) / 6;

  // Number of additional Nedelec polynomials that could be added
  const std::size_t ns = degree * (degree + 1) / 2;

  // Number of polynomials that would be included that are not
  // independent so are removed
  const std::size_t ns_remove = degree * (degree - 1) / 2;

  // Number of dofs in the space, ie size of polynomial set
  const std::size_t ndofs = 6 * degree + 4 * degree * (degree - 1)
                            + (degree - 2) * (degree - 1) * degree / 2;

  // Tabulate polynomial basis at quadrature points
  const auto [_pts, wts] = quadrature::make_quadrature<T>(
      quadrature::type::Default, cell::type::tetrahedron,
      polyset::type::standard, 2 * degree);
  impl::mdspan_t<const T, 2> pts(_pts.data(), wts.size(),
                                 _pts.size() / wts.size());
  const auto [_phi, shape] = polyset::tabulate(
      cell::type::tetrahedron, polyset::type::standard, degree, 0, pts);
  impl::mdspan_t<const T, 3> phi(_phi.data(), shape);
  const std::size_t psize = phi.extent(1);

  // Create coefficients for order (degree-1) polynomials
  impl::mdarray_t<T, 2> wcoeffs(ndofs, psize * tdim);
  for (std::size_t i = 0; i < tdim; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      wcoeffs(i * nv + j, i * psize + j) = 1.0;

  // Create coefficients for additional Nedelec polynomials.
  //
  // Each of the three loops below originally computed, for every (i, j),
  // a sum over quadrature points k of
  //   wts[k] * phi(0,ns0+i,k) * pts(k,dim) * phi(0,j,k)
  // via an explicit triple loop, for dim = 2, 1, 0 respectively. Since
  // phi(0,j,k) (for j in [nv,psize)) does not depend on i or dim, and the
  // wts[k]*pts(k,dim)*phi(0,ns0+i,k) factor does not depend on j, these
  // are three matrix-matrix products, computed here via BLAS (math::dot)
  // instead; the row/column targets and signs below are unchanged from
  // the three original loops.
  const std::size_t npts = wts.size();
  const std::size_t ncols = psize - nv;

  // B(k, jc) = phi(0, nv + jc, k)
  std::vector<T> Bb(npts * ncols);
  impl::mdspan_t<T, 2> B(Bb.data(), npts, ncols);
  for (std::size_t k = 0; k < npts; ++k)
    for (std::size_t jc = 0; jc < ncols; ++jc)
      B(k, jc) = phi(0, nv + jc, k);

  // A[dim](i, k) = wts[k] * pts(k, dim) * phi(0, ns0 + i, k)
  std::array<std::vector<T>, 3> Ab{std::vector<T>(ns * npts),
                                   std::vector<T>(ns * npts),
                                   std::vector<T>(ns * npts)};
  std::array<impl::mdspan_t<T, 2>, 3> A{
      impl::mdspan_t<T, 2>(Ab[0].data(), ns, npts),
      impl::mdspan_t<T, 2>(Ab[1].data(), ns, npts),
      impl::mdspan_t<T, 2>(Ab[2].data(), ns, npts)};
  for (std::size_t i = 0; i < ns; ++i)
    for (std::size_t k = 0; k < npts; ++k)
    {
      const T p = wts[k] * phi(0, ns0 + i, k);
      A[0](i, k) = p * pts(k, 0);
      A[1](i, k) = p * pts(k, 1);
      A[2](i, k) = p * pts(k, 2);
    }

  // W[dim] = A[dim] * B
  std::array<std::vector<T>, 3> Wb{std::vector<T>(ns * ncols),
                                   std::vector<T>(ns * ncols),
                                   std::vector<T>(ns * ncols)};
  std::array<impl::mdspan_t<T, 2>, 3> W{
      impl::mdspan_t<T, 2>(Wb[0].data(), ns, ncols),
      impl::mdspan_t<T, 2>(Wb[1].data(), ns, ncols),
      impl::mdspan_t<T, 2>(Wb[2].data(), ns, ncols)};
  for (int dim = 0; dim < 3; ++dim)
    math::dot(A[dim], B, W[dim]);

  for (std::size_t i = 0; i < ns; ++i)
  {
    for (std::size_t jc = 0; jc < ncols; ++jc)
    {
      const std::size_t j = nv + jc;
      const T w0 = W[0](i, jc);
      const T w1 = W[1](i, jc);
      const T w2 = W[2](i, jc);

      // dim = 2 (first original loop)
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize + j) = -w2;
      wcoeffs(tdim * nv + i + ns - ns_remove, j) = w2;

      // dim = 1 (second original loop)
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, j) = -w1;
      if (i >= ns_remove)
        wcoeffs(tdim * nv + i - ns_remove, psize * 2 + j) = w1;

      // dim = 0 (third original loop)
      wcoeffs(tdim * nv + i + ns - ns_remove, psize * 2 + j) = -w0;
      wcoeffs(tdim * nv + i + ns * 2 - ns_remove, psize + j) = w0;
    }
  }

  math::orthogonalise<T>(wcoeffs, nv * 3);

  return wcoeffs;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> element::create_nedelec(cell::type celltype, int degree,
                                         lagrange_variant lvariant,
                                         bool discontinuous)
{
  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  const std::size_t tdim = cell::topological_dimension(celltype);

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 0);
    x[0] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[0] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  std::vector<T> wcoeffs;
  std::array<std::size_t, 2> wshape;
  switch (celltype)
  {
  case cell::type::triangle:
  {
    impl::mdarray_t<T, 2> w = create_nedelec_2d_space<T>(degree);
    wshape = {w.extent(0), w.extent(1)};
    wcoeffs.resize(wshape[0] * wshape[1]);
    std::copy_n(w.data(), w.size(), wcoeffs.data());
    break;
  }
  case cell::type::tetrahedron:
  {
    impl::mdarray_t<T, 2> w = create_nedelec_3d_space<T>(degree);
    wshape = {w.extent(0), w.extent(1)};
    wcoeffs.resize(wshape[0] * wshape[1]);
    std::copy_n(w.data(), w.size(), wcoeffs.data());
    break;
  }
  default:
    throw std::runtime_error("Invalid celltype in Nedelec");
  }

  // Integral representation for the boundary (edge) dofs
  {
    FiniteElement edge_space = element::create_lagrange<T>(
        cell::type::interval, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments<T>(
        edge_space, celltype, polyset::type::standard, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[1].emplace_back(std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]},
                        _M[i]);
    }
  }

  // Face dofs
  if (degree > 1)
  {
    FiniteElement face_space = element::create_lagrange<T>(
        cell::type::triangle, degree - 2, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_integral_moments<T>(
        face_space, celltype, polyset::type::standard, tdim, 2 * degree - 2);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[2].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[2].emplace_back(std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]},
                        _M[i]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[2] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  // Volume dofs
  if (tdim == 3)
  {
    if (degree > 2 and tdim == 3)
    {
      auto [_x, xshape, _M, Mshape] = moments::make_integral_moments<T>(
          element::create_lagrange<T>(cell::type::tetrahedron, degree - 3,
                                      lvariant, true),
          cell::type::tetrahedron, polyset::type::standard, 3, 2 * degree - 3);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
        M[3].emplace_back(
            std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]}, _M[i]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
    }
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    // std::tie(x, M) = element::make_discontinuous(x, M, tdim, tdim);
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  sobolev::space space
      = discontinuous ? sobolev::space::L2 : sobolev::space::HCurl;
  return FiniteElement<T>(
      element::family::N1E, celltype, polyset::type::standard, degree, {tdim},
      impl::mdspan_t<T, 2>(wcoeffs.data(), wshape), xview, Mview, 0,
      maps::type::covariantPiola, space, discontinuous, degree - 1, degree,
      lvariant, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T> element::create_nedelec2(cell::type celltype, int degree,
                                          lagrange_variant lvariant,
                                          bool discontinuous)
{
  if (celltype != cell::type::triangle and celltype != cell::type::tetrahedron)
    throw std::runtime_error("Invalid celltype in Nedelec");

  if (degree < 1)
    throw std::runtime_error("Degree must be at least 1");

  std::array<std::vector<impl::mdarray_t<T, 2>>, 4> x;
  std::array<std::vector<impl::mdarray_t<T, 4>>, 4> M;

  const std::size_t tdim = cell::topological_dimension(celltype);
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 0);
    x[0] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[0] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  // Integral representation for the edge dofs

  {
    FiniteElement edge_space = element::create_lagrange<T>(
        cell::type::interval, degree, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_tangent_integral_moments<T>(
        edge_space, celltype, polyset::type::standard, tdim, 2 * degree);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[1].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[1].emplace_back(std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]},
                        _M[i]);
    }
  }

  if (degree > 1)
  {
    // Integral moments on faces
    FiniteElement face_space = element::create_rt<T>(
        cell::type::triangle, degree - 1, lvariant, true);
    auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<T>(
        face_space, celltype, polyset::type::standard, tdim, 2 * degree - 1);
    assert(_x.size() == _M.size());
    for (std::size_t i = 0; i < _x.size(); ++i)
    {
      x[2].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
      M[2].emplace_back(std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]},
                        _M[i]);
    }
  }
  else
  {
    const std::size_t num_ent = cell::num_sub_entities(celltype, 2);
    x[2] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
    M[2] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
  }

  if (tdim == 3)
  {
    if (degree > 2)
    {
      auto [_x, xshape, _M, Mshape] = moments::make_dot_integral_moments<T>(
          element::create_rt<T>(cell::type::tetrahedron, degree - 2, lvariant,
                                true),
          celltype, polyset::type::standard, tdim, 2 * degree - 2);
      assert(_x.size() == _M.size());
      for (std::size_t i = 0; i < _x.size(); ++i)
      {
        x[3].emplace_back(std::array{xshape[0], xshape[1]}, _x[i]);
        M[3].emplace_back(
            std::array{Mshape[0], Mshape[1], Mshape[2], Mshape[3]}, _M[i]);
      }
    }
    else
    {
      const std::size_t num_ent = cell::num_sub_entities(celltype, 3);
      x[3] = std::vector(num_ent, impl::mdarray_t<T, 2>(0, tdim));
      M[3] = std::vector(num_ent, impl::mdarray_t<T, 4>(0, tdim, 0, 1));
    }
  }

  std::array<std::vector<mdspan_t<const T, 2>>, 4> xview = impl::to_mdspan(x);
  std::array<std::vector<mdspan_t<const T, 4>>, 4> Mview = impl::to_mdspan(M);
  std::array<std::vector<std::vector<T>>, 4> xbuffer;
  std::array<std::vector<std::vector<T>>, 4> Mbuffer;
  if (discontinuous)
  {
    std::array<std::vector<std::array<std::size_t, 2>>, 4> xshape;
    std::array<std::vector<std::array<std::size_t, 4>>, 4> Mshape;
    std::tie(xbuffer, xshape, Mbuffer, Mshape)
        = element::make_discontinuous(xview, Mview, tdim, tdim);
    xview = impl::to_mdspan(xbuffer, xshape);
    Mview = impl::to_mdspan(Mbuffer, Mshape);
  }

  const std::size_t psize
      = polyset::dim(celltype, polyset::type::standard, degree);
  return FiniteElement<T>(
      element::family::N2E, celltype, polyset::type::standard, degree, {tdim},
      impl::mdspan_t<T, 2>(math::eye<T>(tdim * psize).data(), tdim * psize,
                           tdim * psize),
      xview, Mview, 0, maps::type::covariantPiola, sobolev::space::HCurl,
      discontinuous, degree, degree, lvariant, element::dpc_variant::unset);
}
//-----------------------------------------------------------------------------
/// @cond DOXYGEN_SHOULD_SKIP_THIS
template FiniteElement<float> element::create_nedelec(cell::type, int,
                                                      lagrange_variant, bool);
template FiniteElement<double> element::create_nedelec(cell::type, int,
                                                       lagrange_variant, bool);

template FiniteElement<float> element::create_nedelec2(cell::type, int,
                                                       lagrange_variant, bool);
template FiniteElement<double> element::create_nedelec2(cell::type, int,
                                                        lagrange_variant, bool);
/// @endcond
//-----------------------------------------------------------------------------
