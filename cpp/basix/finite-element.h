// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "maps.h"
#include "mdspan.hpp"
#include "precompute.h"
#include <array>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

/// Basix: FEniCS runtime basis evaluation library
namespace basix
{

namespace impl
{
using mdspan2_t
    = std::experimental::mdspan<double,
                                std::experimental::dextents<std::size_t, 2>>;
using mdspan4_t
    = std::experimental::mdspan<double,
                                std::experimental::dextents<std::size_t, 4>>;
using cmdspan2_t
    = std::experimental::mdspan<const double,
                                std::experimental::dextents<std::size_t, 2>>;
using cmdspan3_t
    = std::experimental::mdspan<const double,
                                std::experimental::dextents<std::size_t, 3>>;
using cmdspan4_t
    = std::experimental::mdspan<const double,
                                std::experimental::dextents<std::size_t, 4>>;

using mdarray2_t
    = std::experimental::mdarray<double,
                                 std::experimental::dextents<std::size_t, 2>>;
using mdarray4_t
    = std::experimental::mdarray<double,
                                 std::experimental::dextents<std::size_t, 4>>;

inline std::array<std::vector<cmdspan2_t>, 4>
to_mdspan(std::array<std::vector<mdarray2_t>, 4>& x)
{
  std::array<std::vector<cmdspan2_t>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].emplace_back(x[i][j].data(), x[i][j].extents());

  return x1;
}

inline std::array<std::vector<cmdspan4_t>, 4>
to_mdspan(std::array<std::vector<mdarray4_t>, 4>& M)
{
  std::array<std::vector<cmdspan4_t>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].emplace_back(M[i][j].data(), M[i][j].extents());

  return M1;
}

inline std::array<std::vector<cmdspan2_t>, 4>
to_mdspan(const std::array<std::vector<std::vector<double>>, 4>& x,
          const std::array<std::vector<std::array<std::size_t, 2>>, 4>& shape)
{
  std::array<std::vector<cmdspan2_t>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].push_back(cmdspan2_t(x[i][j].data(), shape[i][j]));

  return x1;
}

inline std::array<std::vector<cmdspan4_t>, 4>
to_mdspan(const std::array<std::vector<std::vector<double>>, 4>& M,
          const std::array<std::vector<std::array<std::size_t, 4>>, 4>& shape)
{
  std::array<std::vector<cmdspan4_t>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].push_back(cmdspan4_t(M[i][j].data(), shape[i][j]));

  return M1;
}

} // namespace impl

namespace element
{
/// Typedef for mdspan
using cmdspan2_t = impl::cmdspan2_t;
/// Typedef for mdspan
using cmdspan4_t = impl::cmdspan4_t;

/// Creates a version of the interpolation points, interpolation
/// matrices and entity transformation that represent a discontinuous
/// version of the element. This discontinuous version will have the
/// same DOFs but they will all be associated with the interior of the
/// reference cell.
/// @param[in] x Interpolation points. Indices are (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index, derivative)
/// @param[in] tdim The topological dimension of the cell the element is
/// defined on
/// @param[in] value_size The value size of the element
/// @return (xdata, xshape, Mdata, Mshape), where the x and M data are
/// for  a discontinuous version of the element (with the same shapes as
/// x and M)
std::tuple<std::array<std::vector<std::vector<double>>, 4>,
           std::array<std::vector<std::array<std::size_t, 2>>, 4>,
           std::array<std::vector<std::vector<double>>, 4>,
           std::array<std::vector<std::array<std::size_t, 4>>, 4>>
make_discontinuous(const std::array<std::vector<cmdspan2_t>, 4>& x,
                   const std::array<std::vector<cmdspan4_t>, 4>& M,
                   std::size_t tdim, std::size_t value_size);

} // namespace element

/// A finite element

/// The basis of a finite element is stored as a set of coefficients,
/// which are applied to the underlying expansion set for that cell
/// type, when tabulating.
class FiniteElement
{
  // namespace stdex = std::experimental;
  using cmdspan2_t
      = std::experimental::mdspan<const double,
                                  std::experimental::dextents<std::size_t, 2>>;
  using cmdspan4_t
      = std::experimental::mdspan<const double,
                                  std::experimental::dextents<std::size_t, 4>>;

public:
  /// A finite element
  ///
  /// Initialising a finite element calculates the basis functions of the finite
  /// element, in terms of the polynomial basis.
  ///
  /// The below explanation uses Einstein notation.
  ///
  /// The basis functions @f${\phi_i}@f$ of a finite element are represented
  /// as a linear combination of polynomials @f$\{p_j\}@f$ in an underlying
  /// polynomial basis that span the space of all d-dimensional polynomials up
  /// to order @f$k \ (P_k^d)@f$:
  /// \f[ \phi_i = c_{ij} p_j \f]
  ///
  /// In some cases, the basis functions @f$\{\phi_i\}@f$ do not span the
  /// full space @f$P_k@f$, in which case we denote space spanned by the
  /// basis functions by @f$\{q_k\}@f$, which can be represented by:
  /// @f[  q_i = b_{ij} p_j. @f]
  ///  This leads to
  /// @f[  \phi_i = c^{\prime}_{ij} q_j = c^{\prime}_{ij} b_{jk} p_k,  @f]
  /// and in matrix form:
  /// \f[
  /// \phi = C^{\prime} B p
  /// \f]
  ///
  /// If the basis functions span the full space, then @f$ B @f$ is simply
  /// the identity.
  ///
  /// The basis functions @f$\phi_i@f$ are defined by a dual set of functionals
  /// @f$\{f_i\}@f$. The basis functions are the functions in span{@f$q_k@f$}
  /// such that
  ///   @f[ f_i(\phi_j) = \delta_{ij} @f]
  /// and inserting the expression for @f$\phi_{j}@f$:
  ///   @f[ f_i(c^{\prime}_{jk}b_{kl}p_{l}) = c^{\prime}_{jk} b_{kl} f_i \left(
  ///   p_{l} \right) @f]
  ///
  /// Defining a matrix D given by applying the functionals to each
  /// polynomial @f$p_j@f$:
  ///  @f[ [D] = d_{ij},\mbox{ where } d_{ij} = f_i(p_j), @f]
  /// we have:
  /// @f[ C^{\prime} B D^{T} = I @f]
  ///
  /// and
  ///
  /// @f[ C^{\prime} = (B D^{T})^{-1}. @f]
  ///
  /// Recalling that @f$C = C^{\prime} B@f$, where @f$C@f$ is the matrix
  /// form of @f$c_{ij}@f$,
  ///
  /// @f[ C = (B D^{T})^{-1} B @f]
  ///
  /// This function takes the matrices @f$B@f$ (`wcoeffs`) and @f$D@f$ (`M`) as
  /// inputs and will internally compute @f$C@f$.
  ///
  /// The matrix @f$BD^{T}@f$ can be obtained from an element by using the
  /// function `dual_matrix()`. The matrix @f$C@f$ can be obtained from an
  /// element by using the function `coefficient_matrix()`.
  ///
  /// Example: Order 1 Lagrange elements on a triangle
  /// ------------------------------------------------
  /// On a triangle, the scalar expansion basis is:
  ///  @f[ p_0 = \sqrt{2}/2 \qquad
  ///   p_1 = \sqrt{3}(2x + y - 1) \qquad
  ///   p_2 = 3y - 1 @f]
  /// These span the space @f$P_1@f$.
  ///
  /// Lagrange order 1 elements span the space P_1, so in this example,
  /// B (span_coeffs) is the identity matrix:
  ///   @f[ B = \begin{bmatrix}
  ///                   1 & 0 & 0 \\
  ///                   0 & 1 & 0 \\
  ///                   0 & 0 & 1 \end{bmatrix} @f]
  ///
  /// The functionals defining the Lagrange order 1 space are point
  /// evaluations at the three vertices of the triangle. The matrix D
  /// (dual) given by applying these to p_0 to p_2 is:
  ///  @f[ \mbox{dual} = \begin{bmatrix}
  ///              \sqrt{2}/2 &  -\sqrt{3} & -1 \\
  ///              \sqrt{2}/2 &   \sqrt{3} & -1 \\
  ///              \sqrt{2}/2 &          0 &  2 \end{bmatrix} @f]
  ///
  /// For this example, this function outputs the matrix:
  ///  @f[ C = \begin{bmatrix}
  ///            \sqrt{2}/3 & -\sqrt{3}/6 &  -1/6 \\
  ///            \sqrt{2}/3 & \sqrt{3}/6  &  -1/6 \\
  ///            \sqrt{2}/3 &          0  &   1/3 \end{bmatrix} @f]
  /// The basis functions of the finite element can be obtained by applying
  /// the matrix C to the vector @f$[p_0, p_1, p_2]@f$, giving:
  ///   @f[ \begin{bmatrix} 1 - x - y \\ x \\ y \end{bmatrix} @f]
  ///
  /// Example: Order 1 Raviart-Thomas on a triangle
  /// ---------------------------------------------
  /// On a triangle, the 2D vector expansion basis is:
  ///  @f[ \begin{matrix}
  ///   p_0 & = & (\sqrt{2}/2, 0) \\
  ///   p_1 & = & (\sqrt{3}(2x + y - 1), 0) \\
  ///   p_2 & = & (3y - 1, 0) \\
  ///   p_3 & = & (0, \sqrt{2}/2) \\
  ///   p_4 & = & (0, \sqrt{3}(2x + y - 1)) \\
  ///   p_5 & = & (0, 3y - 1)
  ///  \end{matrix}
  /// @f]
  /// These span the space @f$ P_1^2 @f$.
  ///
  /// Raviart-Thomas order 1 elements span a space smaller than @f$ P_1^2 @f$,
  /// so B (span_coeffs) is not the identity. It is given by:
  ///   @f[ B = \begin{bmatrix}
  ///  1 &  0 &  0 &    0 &  0 &   0 \\
  ///  0 &  0 &  0 &    1 &  0 &     0 \\
  ///  1/12 &  \sqrt{6}/48 &  -\sqrt{2}/48 &  1/12 &  0 &  \sqrt{2}/24
  ///  \end{bmatrix}
  ///  @f]
  /// Applying the matrix B to the vector @f$[p_0, p_1, ..., p_5]@f$ gives the
  /// basis of the polynomial space for Raviart-Thomas:
  ///   @f[ \begin{bmatrix}
  ///  \sqrt{2}/2 &  0 \\
  ///   0 &  \sqrt{2}/2 \\
  ///   \sqrt{2}x/8  & \sqrt{2}y/8
  ///  \end{bmatrix} @f]
  ///
  /// The functionals defining the Raviart-Thomas order 1 space are integral
  /// of the normal components along each edge. The matrix D (dual) given
  /// by applying these to @f$p_0@f$ to @f$p_5@f$ is:
  /// @f[ D = \begin{bmatrix}
  /// -\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 & -\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 \\
  /// -\sqrt{2}/2 &  \sqrt{3}/2 & -1/2 &          0  &          0 &    0 \\
  ///           0 &         0   &    0 &  \sqrt{2}/2 &          0 &   -1
  /// \end{bmatrix} @f]
  ///
  /// In this example, this function outputs the matrix:
  ///  @f[  C = \begin{bmatrix}
  ///  -\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 & -\sqrt{2}/2 & -\sqrt{3}/2 & -1/2 \\
  ///  -\sqrt{2}/2 &  \sqrt{3}/2 & -1/2 &          0  &          0  &    0 \\
  ///            0 &          0  &    0 &  \sqrt{2}/2 &          0  &   -1
  /// \end{bmatrix} @f]
  /// The basis functions of the finite element can be obtained by applying
  /// the matrix C to the vector @f$[p_0, p_1, ..., p_5]@f$, giving:
  ///   @f[ \begin{bmatrix}
  ///   -x & -y \\
  ///   x - 1 & y \\
  ///   -x & 1 - y \end{bmatrix} @f]
  ///
  /// @param[in] family The element family
  /// @param[in] cell_type The cell type
  /// @param[in] degree The degree of the element
  /// @param[in] interpolation_nderivs The number of derivatives that need to be
  /// used during interpolation
  /// @param[in] value_shape The value shape of the element
  /// @param[in] wcoeffs Matrices for the kth value index containing the
  /// expansion coefficients defining a polynomial basis spanning the
  /// polynomial space for this element. Shape is (dim(Legendre polynomials),
  /// dim(finite element polyset))
  /// @param[in] x Interpolation points. Indices are (tdim, entity index,
  /// point index, dim)
  /// @param[in] M The interpolation matrices. Indices are (tdim, entity
  /// index, dof, vs, point_index, derivative)
  /// @param[in] map_type The type of map to be used to map values from
  /// the reference to a cell
  /// @param[in] discontinuous Indicates whether or not this is the
  /// discontinuous version of the element
  /// @param[in] highest_complete_degree The highest degree n such that a
  /// Lagrange (or vector Lagrange) element of degree n is a subspace of this
  /// element
  /// @param[in] highest_degree The highest degree n such that at least one
  /// polynomial of degree n is included in this element's polymonial set
  /// @param[in] lvariant The Lagrange variant of the element
  /// @param[in] dvariant The DPC variant of the element
  /// @param[in] tensor_factors The factors in the tensor product representation
  /// of this element
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape,
      const xt::xtensor<double, 2>& wcoeffs,
      const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
      const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      element::lagrange_variant lvariant, element::dpc_variant dvariant,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// TODO
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape, const cmdspan2_t& wcoeffs,
      const std::array<std::vector<cmdspan2_t>, 4>& x,
      const std::array<std::vector<cmdspan4_t>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      element::lagrange_variant lvariant, element::dpc_variant dvariant,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// Overload
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape,
      const xt::xtensor<double, 2>& wcoeffs,
      const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
      const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      element::lagrange_variant lvariant,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// TODO
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape, const cmdspan2_t& wcoeffs,
      const std::array<std::vector<cmdspan2_t>, 4>& x,
      const std::array<std::vector<cmdspan4_t>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      element::lagrange_variant lvariant,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// Overload
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape,
      const xt::xtensor<double, 2>& wcoeffs,
      const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
      const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      element::dpc_variant dvariant,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// TODO
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape, const cmdspan2_t& wcoeffs,
      const std::array<std::vector<cmdspan2_t>, 4>& x,
      const std::array<std::vector<cmdspan4_t>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      element::dpc_variant dvariant,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// Overload
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape,
      const xt::xtensor<double, 2>& wcoeffs,
      const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
      const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool discontinuous,
      int highest_complete_degree, int highest_degree,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// TODO
  FiniteElement(
      element::family family, cell::type cell_type, int degree,
      const std::vector<std::size_t>& value_shape, const cmdspan2_t& wcoeffs,
      const std::array<std::vector<cmdspan2_t>, 4>& x,
      const std::array<std::vector<cmdspan4_t>, 4>& M,
      int interpolation_nderivs, maps::type map_type, bool disccontinuous,
      int highest_complete_degree, int highest_degree,
      std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
          tensor_factors
      = {});

  /// Copy constructor
  FiniteElement(const FiniteElement& element) = default;

  /// Move constructor
  FiniteElement(FiniteElement&& element) = default;

  /// Destructor
  ~FiniteElement() = default;

  /// Assignment operator
  FiniteElement& operator=(const FiniteElement& element) = default;

  /// Move assignment operator
  FiniteElement& operator=(FiniteElement&& element) = default;

  /// Check if two elements are the same
  /// @note This operator compares the element properties, eg family,
  /// degree, etc, and not computed numerical data
  /// @return True if elements are the same
  bool operator==(const FiniteElement& e) const;

  /// Array shape for tabulate basis values and derivatives at set of
  /// points.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] num_points Number of points that basis will be computed
  /// at
  /// @return The shape of the array to will filled when passed to
  /// `FiniteElement::tabulate`
  std::array<std::size_t, 4> tabulate_shape(std::size_t nd,
                                            std::size_t num_points) const;

  /// @brief Compute basis values and derivatives at set of points.
  ///
  /// @note The version of `FiniteElement::tabulate` with the basis data
  /// as an out argument should be preferred for repeated call where
  /// performance is critical
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions.
  /// The shape of x is (number of points, geometric dimension).
  /// @return The basis functions (and derivatives). The shape is
  /// (derivative, point, basis fn index, value index).
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, ie for
  /// the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
  /// (0,2), (3,0)... The function basix::indexing::idx can be used to find the
  /// appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  std::pair<std::vector<double>, std::array<std::size_t, 4>>
  tabulate(int nd, impl::cmdspan2_t x) const;

  /// Compute basis values and derivatives at set of points.
  ///
  /// @note The version of `FiniteElement::tabulate` with the basis data
  /// as an out argument should be preferred for repeated call where
  /// performance is critical
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions (row-major
  /// storage).
  /// @param[in] shape The shape `(number of points, geometric
  /// dimension)` of the `x` array.
  /// @return The basis functions (and derivatives). The shape is
  /// (derivative, point, basis fn index, value index).
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, ie for
  /// the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
  /// (0,2), (3,0)... The function basix::indexing::idx can be used to find the
  /// appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  xt::xtensor<double, 4> tabulate(int nd, const xtl::span<const double>& x,
                                  std::array<std::size_t, 2> shape) const;

  /// Compute basis values and derivatives at set of points.
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions.
  /// The shape of x is (number of points, geometric dimension).
  /// @param [out] basis Memory location to fill. It must be allocated
  /// with shape (num_derivatives, num_points, num basis functions,
  /// value_size). The function `FiniteElement::tabulate_shape` can be
  /// used to get the required shape.
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, ie for
  /// the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
  /// (0,2), (3,0)... The function basix::indexing::idx can be used to
  /// find the appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  ///
  /// @todo Remove all internal dynamic memory allocation, pass scratch
  /// space as required
  void tabulate(int nd, const xt::xtensor<double, 2>& x,
                xt::xtensor<double, 4>& basis) const;

  /// Get the element cell type
  /// @return The cell type
  cell::type cell_type() const;

  /// Get the element polynomial degree
  /// @return Polynomial degree
  int degree() const;

  /// Get the lowest degree n such that the highest degree
  /// polynomial in this element is contained in a Lagrange (or vector
  /// Lagrange) element of degree n
  /// @return Polynomial degree
  int highest_degree() const;

  /// Get the highest degree n such that a Lagrange (or vector Lagrange) element
  /// of degree n is a subspace of this element
  /// @return Polynomial degree
  int highest_complete_degree() const;

  /// The element value tensor shape, eg returning {} for scalars, {3}
  /// for vectors in 3D, {2, 2} for a rank-2 tensor in 2D.
  /// @return Value shape
  const std::vector<int>& value_shape() const;

  /// Dimension of the finite element space (number of
  /// degrees-of-freedom for the element)
  /// @return Number of degrees of freedom
  int dim() const;

  /// Get the finite element family
  /// @return The family
  element::family family() const;

  /// Get the Lagrange variant of the element.
  /// @return The Lagrange variant
  element::lagrange_variant lagrange_variant() const;

  /// Get the DPC variant of the element.
  /// @return The DPC variant
  element::dpc_variant dpc_variant() const;

  /// Get the map type for this element
  /// @return The map type
  maps::type map_type() const;

  /// Indicates whether this element is the discontinuous variant
  /// @return True if this element is a discontinuous version of the
  /// element
  bool discontinuous() const;

  /// Indicates whether the dof transformations are all permutations
  /// @return True or False
  bool dof_transformations_are_permutations() const;

  /// Indicates whether the dof transformations are all the identity
  /// @return True or False
  bool dof_transformations_are_identity() const;

  /// Map function values from the reference to a physical cell. This
  /// function can perform the mapping for multiple points, grouped by
  /// points that share a common Jacobian.
  /// @param[in] U The function values on the reference. The indices are
  /// [Jacobian index, point index, components].
  /// @param[in] J The Jacobian of the mapping. The indices are
  /// [Jacobian index, J_i, J_j].
  /// @param[in] detJ The determinant of the Jacobian of the mapping. It
  /// has length `J.shape(0)`
  /// @param[in] K The inverse of the Jacobian of the mapping. The
  /// indices are [Jacobian index, K_i, K_j].
  /// @return The function values on the cell. The indices are [Jacobian
  /// index, point index, components].
  xt::xtensor<double, 3> push_forward(const xt::xtensor<double, 3>& U,
                                      const xt::xtensor<double, 3>& J,
                                      const xtl::span<const double>& detJ,
                                      const xt::xtensor<double, 3>& K) const;

  /// Map function values from a physical cell to the reference
  /// @param[in] u The function values on the cell
  /// @param[in] J The Jacobian of the mapping
  /// @param[in] detJ The determinant of the Jacobian of the mapping
  /// @param[in] K The inverse of the Jacobian of the mapping
  /// @return The function values on the reference. The indices are [Jacobian
  /// index, point index, components].
  xt::xtensor<double, 3> pull_back(const xt::xtensor<double, 3>& u,
                                   const xt::xtensor<double, 3>& J,
                                   const xtl::span<const double>& detJ,
                                   const xt::xtensor<double, 3>& K) const;

  /// Return a function that performs the appropriate
  /// push-forward/pull-back for the element type
  ///
  /// @tparam O The type that hold the (computed) mapped data (ndim==2)
  /// @tparam P The type that hold the data to be mapped (ndim==2)
  /// @tparam Q The type that holds the Jacobian (or inverse) matrix (ndim==2)
  /// @tparam R The type that holds the inverse of the `Q` data
  /// (ndim==2)
  ///
  /// @return A function that for a push-forward takes arguments
  /// - `u` [out] The data on the physical cell after the
  /// push-forward flattened with row-major layout, shape=(num_points,
  /// value_size)
  /// - `U` [in] The data on the reference cell physical field to push
  /// forward, flattened with row-major layout, shape=(num_points,
  /// ref_value_size)
  /// - `J` [in] The Jacobian matrix of the map ,shape=(gdim, tdim)
  /// - `detJ` [in] det(J)
  /// - `K` [in] The inverse of the Jacobian matrix, shape=(tdim, gdim)
  ///
  /// For a pull-back the arguments should be:
  /// - `U` [out] The data on the reference cell after the pull-back,
  /// flattened with row-major layout, shape=(num_points, ref
  /// value_size)
  /// - `u` [in] The data on the physical cell that should be pulled
  /// back , flattened with row-major layout, shape=(num_points,
  /// value_size)
  /// - `K` [in] The inverse of the Jacobian matrix of the map
  /// ,shape=(tdim, gdim)
  /// - `detJ_inv` [in] 1/det(J)
  /// - `J` [in] The Jacobian matrix, shape=(gdim, tdim)
  template <typename O, typename P, typename Q, typename R>
  std::function<void(O&, const P&, const Q&, double, const R&)> map_fn() const
  {
    switch (_map_type)
    {
    case maps::type::identity:
      return [](O& u, const P& U, const Q&, double, const R&)
      {
        // std::cout << "ident" << std::endl;
        assert(U.extent(0) == u.extent(0));
        assert(U.extent(1) == u.extent(1));
        for (std::size_t i = 0; i < U.extent(0); ++i)
          for (std::size_t j = 0; j < U.extent(1); ++j)
            u(i, j) = U(i, j);
      };
    case maps::type::covariantPiola:
      return [](O& u, const P& U, const Q& J, double detJ, const R& K)
      {
        // std::cout << "map 1" << std::endl;
        maps::covariant_piola(u, U, J, detJ, K);
      };
    case maps::type::contravariantPiola:
      return [](O& u, const P& U, const Q& J, double detJ, const R& K)
      {
        // std::cout << "map 2" << std::endl;
        maps::contravariant_piola(u, U, J, detJ, K);
      };
    case maps::type::doubleCovariantPiola:
      return [](O& u, const P& U, const Q& J, double detJ, const R& K)
      {
        // std::cout << "map 3" << std::endl;
        maps::double_covariant_piola(u, U, J, detJ, K);
      };
    case maps::type::doubleContravariantPiola:
      return [](O& u, const P& U, const Q& J, double detJ, const R& K)
      {
        // std::cout << "map 4" << std::endl;
        maps::double_contravariant_piola(u, U, J, detJ, K);
      };
    default:
      throw std::runtime_error("Map not implemented");
    }
  }

  /// Get the number of dofs on each topological entity: (vertices,
  /// edges, faces, cell) in that order. For example, Lagrange degree 2
  /// on a triangle has vertices: [1, 1, 1], edges: [1, 1, 1], cell: [0]
  /// The sum of the entity dofs must match the total number of dofs
  /// reported by FiniteElement::dim,
  /// @code{.cpp}
  /// const std::vector<std::vector<int>>& dofs = e.entity_dofs();
  /// int num_dofs0 = dofs[1][3]; // Number of dofs associated with edge 3
  /// int num_dofs1 = dofs[2][0]; // Number of dofs associated with face 0
  /// @endcode
  /// @return Number of dofs associated with an entity of a given
  /// topological dimension. The shape is (tdim + 1, num_entities).
  const std::vector<std::vector<int>>& num_entity_dofs() const;

  /// Get the number of dofs on the closure of each topological entity:
  /// (vertices, edges, faces, cell) in that order. For example, Lagrange degree
  /// 2 on a triangle has vertices: [1, 1, 1], edges: [3, 3, 3], cell: [6]
  /// @return Number of dofs associated with the closure of an entity of a given
  /// topological dimension. The shape is (tdim + 1, num_entities).
  const std::vector<std::vector<int>>& num_entity_closure_dofs() const;

  /// Get the dofs on each topological entity: (vertices,
  /// edges, faces, cell) in that order. For example, Lagrange degree 2
  /// on a triangle has vertices: [[0], [1], [2]], edges: [[3], [4], [5]],
  /// cell: [[]]
  /// @return Dofs associated with an entity of a given
  /// topological dimension. The shape is (tdim + 1, num_entities, num_dofs).
  const std::vector<std::vector<std::vector<int>>>& entity_dofs() const;

  /// Get the dofs on the closure of each topological entity: (vertices,
  /// edges, faces, cell) in that order. For example, Lagrange degree 2
  /// on a triangle has vertices: [[0], [1], [2]], edges: [[1, 2, 3], [0, 2, 4],
  /// [0, 1, 5]], cell: [[0, 1, 2, 3, 4, 5]]
  /// @return Dofs associated with the closure of an entity of a given
  /// topological dimension. The shape is (tdim + 1, num_entities, num_dofs).
  const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs() const;

  /// Get the base transformations
  /// The base transformations represent the effect of rotating or reflecting
  /// a subentity of the cell on the numbering and orientation of the DOFs.
  /// This returns a list of matrices with one matrix for each subentity
  /// permutation in the following order:
  ///   Reversing edge 0, reversing edge 1, ...
  ///   Rotate face 0, reflect face 0, rotate face 1, reflect face 1, ...
  ///
  /// Example: Order 3 Lagrange on a triangle
  /// ---------------------------------------
  /// This space has 10 dofs arranged like:
  /// ~~~~~~~~~~~~~~~~
  /// 2
  /// |\
  /// 6 4
  /// |  \
  /// 5 9 3
  /// |    \
  /// 0-7-8-1
  /// ~~~~~~~~~~~~~~~~
  /// For this element, the base transformations are:
  ///   [Matrix swapping 3 and 4,
  ///    Matrix swapping 5 and 6,
  ///    Matrix swapping 7 and 8]
  /// The first row shows the effect of reversing the diagonal edge. The
  /// second row shows the effect of reversing the vertical edge. The third
  /// row shows the effect of reversing the horizontal edge.
  ///
  /// Example: Order 1 Raviart-Thomas on a triangle
  /// ---------------------------------------------
  /// This space has 3 dofs arranged like:
  /// ~~~~~~~~~~~~~~~~
  ///   |\
  ///   | \
  ///   |  \
  /// <-1   0
  ///   |  / \
  ///   | L ^ \
  ///   |   |  \
  ///    ---2---
  /// ~~~~~~~~~~~~~~~~
  /// These DOFs are integrals of normal components over the edges: DOFs 0 and 2
  /// are oriented inward, DOF 1 is oriented outwards.
  /// For this element, the base transformation matrices are:
  /// ~~~~~~~~~~~~~~~~
  ///   0: [[-1, 0, 0],
  ///       [ 0, 1, 0],
  ///       [ 0, 0, 1]]
  ///   1: [[1,  0, 0],
  ///       [0, -1, 0],
  ///       [0,  0, 1]]
  ///   2: [[1, 0,  0],
  ///       [0, 1,  0],
  ///       [0, 0, -1]]
  /// ~~~~~~~~~~~~~~~~
  /// The first matrix reverses DOF 0 (as this is on the first edge). The second
  /// matrix reverses DOF 1 (as this is on the second edge). The third matrix
  /// reverses DOF 2 (as this is on the third edge).
  ///
  /// Example: DOFs on the face of Order 2 Nedelec first kind on a tetrahedron
  /// ------------------------------------------------------------------------
  /// On a face of this tetrahedron, this space has two face tangent DOFs:
  /// ~~~~~~~~~~~~~~~~
  /// |\        |\
  /// | \       | \
  /// |  \      | ^\
  /// |   \     | | \
  /// | 0->\    | 1  \
  /// |     \   |     \
  ///  ------    ------
  /// ~~~~~~~~~~~~~~~~
  /// For these DOFs, the subblocks of the base transformation matrices are:
  /// ~~~~~~~~~~~~~~~~
  ///   rotation: [[-1, 1],
  ///              [ 1, 0]]
  ///   reflection: [[0, 1],
  ///                [1, 0]]
  /// ~~~~~~~~~~~~~~~~
  /// @return The base transformations for this element. The shape is
  /// (ntranformations, ndofs, ndofs)
  xt::xtensor<double, 3> base_transformations() const;

  /// Return the entity dof transformation matrices
  /// @return The entity transformations for the subentities of this element.
  /// The shape for each cell is (ntransformations, ndofs, ndofs)
  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations() const;

  /// Permute the dof numbering on a cell
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] dofs The dof numbering for the cell
  /// @param cell_info The permutation info for the cell
  void permute_dofs(const xtl::span<std::int32_t>& dofs,
                    std::uint32_t cell_info) const;

  /// Unpermute the dof numbering on a cell
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] dofs The dof numbering for the cell
  /// @param cell_info The permutation info for the cell
  void unpermute_dofs(const xtl::span<std::int32_t>& dofs,
                      std::uint32_t cell_info) const;

  /// Apply DOF transformations to some data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_dof_transformation(const xtl::span<T>& data, int block_size,
                                std::uint32_t cell_info) const;

  /// Apply transpose DOF transformations to some data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_transpose_dof_transformation(const xtl::span<T>& data,
                                          int block_size,
                                          std::uint32_t cell_info) const;

  /// Apply inverse transpose DOF transformations to some data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_transpose_dof_transformation(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Apply inverse DOF transformations to some data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_dof_transformation(const xtl::span<T>& data,
                                        int block_size,
                                        std::uint32_t cell_info) const;

  /// Apply DOF transformations to some transposed data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_dof_transformation_to_transpose(const xtl::span<T>& data,
                                             int block_size,
                                             std::uint32_t cell_info) const;

  /// Apply transpose DOF transformations to some transposed data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_transpose_dof_transformation_to_transpose(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Apply inverse transpose DOF transformations to some transposed data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_transpose_dof_transformation_to_transpose(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Apply inverse DOF transformations to some transposed data
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_dof_transformation_to_transpose(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Return the interpolation points, i.e. the coordinates on the
  /// reference element where a function need to be evaluated in order
  /// to interpolate it in the finite element space. @return Array of
  /// coordinate with shape `(num_points, tdim)`
  const xt::xtensor<double, 2>& points() const;

  /// Return the interpolation points, ie the coordinates on the
  /// reference element where a function need to be evaluated in order
  /// to interpolate it in the finite element space.
  /// @return Array of coordinate with shape `(num_points, tdim)`
  std::pair<std::vector<double>, std::array<std::size_t, 2>> points_new() const;

  /// @brief Return a matrix of weights interpolation,
  ///
  /// To interpolate a function in this finite element, the functions
  /// should be evaluated at each point given by
  /// FiniteElement::points(). These function values should then be
  /// multiplied by the weight matrix to give the coefficients of the
  /// interpolated function.
  ///
  /// The shape of the returned matrix will be `(dim, num_points *
  /// value_size)`, where `dim` is the number of DOFs in the finite
  /// element, `num_points` is the number of points returned by
  /// `points()`, and `value_size` is the value size of the finite
  /// element.
  ///
  /// For example, to interpolate into a Lagrange space, the following
  /// should be done:
  /// \code{.pseudo}
  /// i_m = element.interpolation_matrix()
  /// pts = element.points()
  /// values = vector(pts.shape(0))
  /// FOR i, p IN ENUMERATE(pts):
  ///     values[i] = f.evaluate_at(p)
  /// coefficients = i_m * values
  /// \endcode
  ///
  /// To interpolate into a Raviart-Thomas space, the following should
  /// be done:
  /// \code{.pseudo}
  /// i_m = element.interpolation_matrix()
  /// pts = element.points()
  /// vs = prod(element.value_shape())
  /// values = VECTOR(pts.shape(0) * vs)
  /// FOR i, p IN ENUMERATE(pts):
  ///     values[i::pts.shape(0)] = f.evaluate_at(p)
  /// coefficients = i_m * values
  /// \endcode
  ///
  /// To interpolate into a Lagrange space with a block size, the
  /// following should be done:
  /// \code{.pseudo}
  /// i_m = element.interpolation_matrix()
  /// pts = element.points()
  /// coefficients = VECTOR(element.dim() * block_size)
  /// FOR b IN RANGE(block_size):
  ///     values = vector(pts.shape(0))
  ///     FOR i, p IN ENUMERATE(pts):
  ///         values[i] = f.evaluate_at(p)[b]
  ///     coefficients[::block_size] = i_m * values
  /// \endcode
  ///
  /// @return The interpolation matrix. Shape is (ndofs, number of interpolation
  /// points)
  const xt::xtensor<double, 2>& interpolation_matrix() const;

  /// @brief Return a matrix of weights interpolation,
  ///
  /// To interpolate a function in this finite element, the functions
  /// should be evaluated at each point given by
  /// FiniteElement::points(). These function values should then be
  /// multiplied by the weight matrix to give the coefficients of the
  /// interpolated function.
  ///
  /// The shape of the returned matrix will be `(dim, num_points *
  /// value_size)`, where `dim` is the number of DOFs in the finite
  /// element, `num_points` is the number of points returned by
  /// `points()`, and `value_size` is the value size of the finite
  /// element.
  ///
  /// For example, to interpolate into a Lagrange space, the following
  /// should be done:
  /// \code{.pseudo}
  /// i_m = element.interpolation_matrix()
  /// pts = element.points()
  /// values = vector(pts.shape(0))
  /// FOR i, p IN ENUMERATE(pts):
  ///     values[i] = f.evaluate_at(p)
  /// coefficients = i_m * values
  /// \endcode
  ///
  /// To interpolate into a Raviart-Thomas space, the following should
  /// be done:
  /// \code{.pseudo}
  /// i_m = element.interpolation_matrix()
  /// pts = element.points()
  /// vs = prod(element.value_shape())
  /// values = VECTOR(pts.shape(0) * vs)
  /// FOR i, p IN ENUMERATE(pts):
  ///     values[i::pts.shape(0)] = f.evaluate_at(p)
  /// coefficients = i_m * values
  /// \endcode
  ///
  /// To interpolate into a Lagrange space with a block size, the
  /// following should be done:
  /// \code{.pseudo}
  /// i_m = element.interpolation_matrix()
  /// pts = element.points()
  /// coefficients = VECTOR(element.dim() * block_size)
  /// FOR b IN RANGE(block_size):
  ///     values = vector(pts.shape(0))
  ///     FOR i, p IN ENUMERATE(pts):
  ///         values[i] = f.evaluate_at(p)[b]
  ///     coefficients[::block_size] = i_m * values
  /// \endcode
  ///
  /// @return The interpolation matrix. Shape is (ndofs, number of interpolation
  /// points)
  std::pair<std::vector<double>, std::array<std::size_t, 2>>
  interpolation_matrix_new() const;

  /// Get the dual matrix.
  ///
  /// This is the matrix @f$BD^{T}@f$, as described in the documentation
  /// of the `FiniteElement()` constructor.
  /// @return The dual matrix. Shape is (ndofs, ndofs)
  const xt::xtensor<double, 2>& dual_matrix() const;

  /// Get the coefficients that define the polynomial set in terms of the
  /// orthonormal polynomials.
  ///
  /// The polynomials spanned by each finite element in Basix are represented as
  /// a linear combination of the orthonormal polynomials of a given degree on
  /// the cell. Each row of this matrix defines a polynomial in the set spanned
  /// by the finite element.
  ///
  /// For example, the orthonormal polynomials of degree <= 1 on a triangle are
  /// (where a, b, c, d are some constants):
  ///
  ///  - (sqrt(2), 0)
  ///  - (a*x - b, 0)
  ///  - (c*y - d, 0)
  ///  - (0, sqrt(2))
  ///  - (0, a*x - b)
  ///  - (0, c*y - d)
  ///
  /// For a degree 1 Raviart-Thomas element, the first two rows of wcoeffs would
  /// be the following, as (1, 0) and (0, 1) are spanned by the element
  ///
  ///  - [1, 0, 0, 0, 0, 0]
  ///  - [0, 0, 0, 1, 0, 0]
  ///
  /// The third row of wcoeffs in this example would give coefficients that
  /// represent (x, y) in terms of the orthonormal polynomials:
  ///
  ///  - [-b/(a*sqrt(2)), 1/a, 0, -d/(c*sqrt(2)), 0, 1/c]
  ///
  /// These coefficients are only stored for custom elements. This function will
  /// throw an exception if called on a non-custom element
  /// @return Coefficient matrix. Shape is (dim(Lagrange polynomials),
  /// dim(finite element polyset))
  const xt::xtensor<double, 2>& wcoeffs() const;

  /// Get the interpolation points for each subentity.
  ///
  /// The indices of this data are (tdim, entity index, point index, dim).
  const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x() const;

  /// Get the interpolation matrices for each subentity.
  ///
  /// The shape of this data is (tdim, entity index, dof, value size,
  /// point_index, derivative).
  ///
  /// These matrices define how to evaluate the DOF functionals accociated with
  /// each sub-entity of the cell. Given a function f, the functionals
  /// associated with the `e`-th entity of dimension `d` can be computed as
  /// follows:
  ///
  /// \code{.pseudo}
  /// matrix = element.M()[d][e]
  /// pts = element.x()[d][e]
  /// nderivs = element
  /// values = f.eval_derivs(nderivs, pts)
  /// result = ZEROS(matrix.shape(0))
  /// FOR i IN RANGE(matrix.shape(0)):
  ///     FOR j IN RANGE(matrix.shape(1)):
  ///         FOR k IN RANGE(matrix.shape(2)):
  ///             FOR l IN RANGE(matrix.shape(3)):
  ///                 result[i] += matrix[i, j, k, l] * values[l][k][j]
  /// \endcode
  ///
  /// For example, for a degree 1 Raviart-Thomas (RT) element on a triangle, the
  /// DOF functionals are integrals over the edges of the dot product of the
  /// function with the normal to the edge. In this case, `x()` would contain
  /// quadrature points for each edge, and `M()` would by a 1 by 2 by `npoints`
  /// by 1 array for each edge. For each point, the `[0, :, point, 0]` slice of
  /// this would be the quadrature weight multiplied by the normal. For all
  /// entities that are not edges, the entries in `x()` and `M()` for a degree 1
  /// RT element would have size 0.
  ///
  /// These matrices are only stored for custom elements. This function will
  /// throw an exception if called on a non-custom element
  /// @return The interpolation matrices. The indices of this data are (tdim,
  /// entity index, dof, vs, point_index, derivative)
  const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M() const;

  /// Get the matrix of coefficients.
  ///
  /// This is the matrix @f$C@f$, as described in the documentation of
  /// the `FiniteElement()` constructor.
  /// @return The coefficient matrix. Shape is (ndofs, ndofs)
  const xt::xtensor<double, 2>& coefficient_matrix() const;

  /// Indicates whether or not this element has a tensor product
  /// representation.
  bool has_tensor_product_factorisation() const;

  /// Get the tensor product representation of this element, or throw an
  /// error if no such factorisation exists.
  ///
  /// The tensor product representation will be a vector of tuples. Each
  /// tuple contains a vector of finite elements, and a vector on
  /// integers. The vector of finite elements gives the elements on an
  /// interval that appear in the tensor product representation. The
  /// vector of integers gives the permutation between the numbering of
  /// the tensor product DOFs and the number of the DOFs of this Basix
  /// element.
  /// @return The tensor product representation
  std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
  get_tensor_product_representation() const;

  /// Indicates whether or not the interpolation matrix for this element is an
  /// identity matrix
  bool interpolation_is_identity() const;

  /// The number of derivatives needed when interpolating
  int interpolation_nderivs() const;

private:
  // Cell type
  cell::type _cell_type;

  // Topological dimension of the cell
  std::size_t _cell_tdim;

  // Topological dimension of the cell
  std::vector<std::vector<cell::type>> _cell_subentity_types;

  // Finite element family
  element::family _family;

  // Lagrange variant
  element::lagrange_variant _lagrange_variant;

  // Lagrange variant
  element::dpc_variant _dpc_variant;

  // Degree that was input when creating the element
  int _degree;

  // Degree
  int _interpolation_nderivs;

  // Highest degree polynomial in element's polyset
  int _highest_degree;

  // Highest degree space that is a subspace of element's polyset
  int _highest_complete_degree;

  // Value shape
  std::vector<int> _value_shape;

  /// The mapping used to map this element from the reference to a cell
  maps::type _map_type;

  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by @f$\psi_i = \sum_{k} \phi_{k}
  // \alpha^{i}_{k}@f$, then _coeffs(i, j) = @f$\alpha^i_k@f$. ie
  // _coeffs.row(i) are the expansion coefficients for shape function i
  // (@f$\psi_{i}@f$).
  xt::xtensor<double, 2> _coeffs;

  // Number of dofs associated with each cell (sub-)entity
  //
  // The dofs of an element are associated with entities of different
  // topological dimension (vertices, edges, faces, cells). The dofs are
  // listed in this order, with vertex dofs first. Each entry is the dof
  // count on the associated entity, as listed by cell::topology.
  std::vector<std::vector<int>> _num_edofs;

  // Number of dofs associated with the closure of each cell
  // (sub-)entity
  std::vector<std::vector<int>> _num_e_closure_dofs;

  // Dofs associated with each cell (sub-)entity
  std::vector<std::vector<std::vector<int>>> _edofs;

  // Dofs associated with each cell (sub-)entity
  std::vector<std::vector<std::vector<int>>> _e_closure_dofs;

  // Entity transformations
  std::map<cell::type, xt::xtensor<double, 3>> _entity_transformations;
  std::map<cell::type,
           std::pair<std::vector<double>, std::array<std::size_t, 3>>>
      _entity_transformations_new;

  // Set of points used for point evaluation
  // Experimental - currently used for an implementation of
  // "tabulate_dof_coordinates" Most useful for Lagrange. This may change or go
  // away. For non-Lagrange elements, these points will be used in combination
  // with _interpolation_matrix to perform interpolation
  xt::xtensor<double, 2> _points;

  // Interpolation points on the cell. The shape is (entity_dim, num
  // entities of given dimension, num_points, tdim)
  std::array<std::vector<xt::xtensor<double, 2>>, 4> _x;

  /// The interpolation weights and points
  xt::xtensor<double, 2> _matM;

  // Indicates whether or not the DOF transformations are all
  // permutations
  bool _dof_transformations_are_permutations;

  // Indicates whether or not the DOF transformations are all identity
  bool _dof_transformations_are_identity;

  // The entity permutations (factorised). This will only be set if
  // _dof_transformations_are_permutations is True and
  // _dof_transformations_are_identity is False
  std::map<cell::type, std::vector<std::vector<std::size_t>>> _eperm;

  // The reverse entity permutations (factorised). This will only be set
  // if _dof_transformations_are_permutations is True and
  // _dof_transformations_are_identity is False
  std::map<cell::type, std::vector<std::vector<std::size_t>>> _eperm_rev;

  // The entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etrans;
  // std::map<cell::type,
  //          std::vector<std::tuple<
  //              std::vector<std::size_t>, std::vector<double>,
  //              std::pair<std::vector<double>, std::array<std::size_t, 2>>>>>
  //     _etrans_new;

  // The transposed entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etransT;

  // The inverse entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etrans_inv;

  // The inverse transpose entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etrans_invT;

  // Indicates whether or not this is the discontinuous version of the
  // element
  bool _discontinuous;

  // The dual matrix
  xt::xtensor<double, 2> _dual_matrix;

  // Tensor product representation
  // Entries of tuple are (list of elements on an interval, permutation
  // of DOF numbers)
  // @todo: For vector-valued elements, a tensor product type and a
  // scaling factor may additionally be needed.
  std::vector<std::tuple<std::vector<FiniteElement>, std::vector<int>>>
      _tensor_factors;

  // Is the interpolation matrix an identity?
  bool _interpolation_is_identity;

  // The coefficients that define the polynomial set in terms of the orthonormal
  // polynomials
  xt::xtensor<double, 2> _wcoeffs;

  // Interpolation matrices for each entity
  std::array<std::vector<xt::xtensor<double, 4>>, 4> _M;
};

/// Create a custom finite element
/// @param[in] cell_type The cell type
/// @param[in] value_shape The value shape of the element
/// @param[in] wcoeffs Matrices for the kth value index containing the
/// expansion coefficients defining a polynomial basis spanning the
/// polynomial space for this element. Shape is (dim(Legendre polynomials),
/// dim(finite element polyset))
/// @param[in] x Interpolation points. Indices are (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index, derivative)
/// @param[in] interpolation_nderivs The number of derivatives that need to be
/// used during interpolation
/// @param[in] map_type The type of map to be used to map values from
/// the reference to a cell
/// @param[in] discontinuous Indicates whether or not this is the
/// discontinuous version of the element
/// @param[in] highest_complete_degree The highest degree n such that a
/// Lagrange (or vector Lagrange) element of degree n is a subspace of this
/// element
/// @param[in] highest_degree The degree of a polynomial in this element's
/// polyset
/// @return A custom finite element
FiniteElement create_custom_element(
    cell::type cell_type, const std::vector<std::size_t>& value_shape,
    const xt::xtensor<double, 2>& wcoeffs,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 4>>, 4>& M,
    int interpolation_nderivs, maps::type map_type, bool discontinuous,
    int highest_complete_degree, int highest_degree);

/// Create an element using a given Lagrange variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::lagrange_variant lvariant,
                             bool discontinuous);

/// Create an element using a given Lagrange variant and a given DPC variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::lagrange_variant lvariant,
                             element::dpc_variant dvariant, bool discontinuous);

/// Create an element using a given DPC variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::dpc_variant dvariant,
                             bool discontinuous);

/// Create an element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined
/// on
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Indicates whether the element is
/// discontinuous between cells points of the element. The discontinuous
/// element will have the same DOFs, but they will all be associated
/// with the interior of the cell.
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, bool discontinuous);

/// Create a continuous element using a given Lagrange variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined
/// on
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::lagrange_variant lvariant);

/// Create a continuous element using a given Lagrange variant and a given DPC
/// variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined
/// on
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::lagrange_variant lvariant,
                             element::dpc_variant dvariant);

/// Create a continuous element using a given DPC variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined
/// on
/// @param[in] degree The degree of the element
/// @param[in] dvariant The variant of DPC to use
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::dpc_variant dvariant);

/// Create a continuous element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined
/// on
/// @param[in] degree The degree of the element
/// @return A finite element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree);

/// Return the Basix version number
/// @return version string
std::string version();

//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_dof_transformation(const xtl::span<T>& data,
                                             int block_size,
                                             std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etrans.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etrans.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          const auto& m = _etrans.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_transpose_dof_transformation(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etransT.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etransT.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
        {
          const auto& m = _etransT.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_inverse_transpose_dof_transformation(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etrans_invT.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etrans_invT.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          const auto& m = _etrans_invT.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_inverse_dof_transformation(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etrans_inv.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etrans_inv.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
        {
          const auto& m = _etrans_inv.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_dof_transformation_to_transpose(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etrans.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix_to_transpose(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etrans.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          const auto& m = _etrans.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_inverse_transpose_dof_transformation_to_transpose(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etrans_invT.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix_to_transpose(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etrans_invT.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          const auto& m = _etrans_invT.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_transpose_dof_transformation_to_transpose(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etransT.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix_to_transpose(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etransT.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
        {
          const auto& m = _etransT.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::apply_inverse_dof_transformation_to_transpose(
    const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
      {
        const auto& m = _etrans_inv.at(cell::type::interval)[0];
        const auto& v_size_t = std::get<0>(m);
        const auto& v_t = std::get<1>(m);
        const auto& matrix = std::get<2>(m);
        precompute::apply_matrix_to_transpose(
            xtl::span(v_size_t), xtl::span(v_t),
            impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
            data, dofstart, block_size);
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
          const auto& m = _etrans_inv.at(_cell_subentity_types[2][f])[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
        {
          const auto& m = _etrans_inv.at(_cell_subentity_types[2][f])[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& v_t = std::get<1>(m);
          const auto& matrix = std::get<2>(m);
          precompute::apply_matrix_to_transpose(
              xtl::span(v_size_t), xtl::span(v_t),
              impl::cmdspan2_t(matrix.data(), matrix.shape(0), matrix.shape(1)),
              data, dofstart, block_size);
        }
        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------

} // namespace basix
