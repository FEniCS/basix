// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth . Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "maps.h"
#include "mdspan.hpp"
#include "polyset.h"
#include "precompute.h"
#include "sobolev-spaces.h"
#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/// Basix: FEniCS runtime basis evaluation library
namespace basix
{

namespace impl
{
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdarray_t
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::mdarray<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

/// Create a container of cmdspan2_t objects from a container of
/// mdarray2_t objects
template <typename T>
std::array<std::vector<mdspan_t<const T, 2>>, 4>
to_mdspan(std::array<std::vector<mdarray_t<T, 2>>, 4>& x)
{
  std::array<std::vector<mdspan_t<const T, 2>>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].emplace_back(x[i][j].data(), x[i][j].extents());

  return x1;
}

/// Create a container of cmdspan4_t objects from a container of
/// mdarray4_t objects
template <typename T>
std::array<std::vector<mdspan_t<const T, 4>>, 4>
to_mdspan(std::array<std::vector<mdarray_t<T, 4>>, 4>& M)
{
  std::array<std::vector<mdspan_t<const T, 4>>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].emplace_back(M[i][j].data(), M[i][j].extents());

  return M1;
}

/// Create a container of cmdspan2_t objects from containers holding
/// data buffers and shapes
template <typename T>
std::array<std::vector<mdspan_t<const T, 2>>, 4>
to_mdspan(const std::array<std::vector<std::vector<T>>, 4>& x,
          const std::array<std::vector<std::array<std::size_t, 2>>, 4>& shape)
{
  std::array<std::vector<mdspan_t<const T, 2>>, 4> x1;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < x[i].size(); ++j)
      x1[i].push_back(mdspan_t<const T, 2>(x[i][j].data(), shape[i][j]));

  return x1;
}

/// Create a container of cmdspan4_t objects from containers holding
/// data buffers and shapes
template <typename T>
std::array<std::vector<mdspan_t<const T, 4>>, 4>
to_mdspan(const std::array<std::vector<std::vector<T>>, 4>& M,
          const std::array<std::vector<std::array<std::size_t, 4>>, 4>& shape)
{
  std::array<std::vector<mdspan_t<const T, 4>>, 4> M1;
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < M[i].size(); ++j)
      M1[i].push_back(mdspan_t<const T, 4>(M[i][j].data(), shape[i][j]));

  return M1;
}

} // namespace impl

namespace element
{
/// Typedef for mdspan
template <typename T, std::size_t d>
using mdspan_t = impl::mdspan_t<T, d>;

/// Create a version of the interpolation points, interpolation
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
template <std::floating_point T>
std::tuple<std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 2>>, 4>,
           std::array<std::vector<std::vector<T>>, 4>,
           std::array<std::vector<std::array<std::size_t, 4>>, 4>>
make_discontinuous(const std::array<std::vector<mdspan_t<const T, 2>>, 4>& x,
                   const std::array<std::vector<mdspan_t<const T, 4>>, 4>& M,
                   std::size_t tdim, std::size_t value_size);

} // namespace element

/// @brief A finite element.
///
/// The basis of a finite element is stored as a set of coefficients,
/// which are applied to the underlying expansion set for that cell
/// type, when tabulating.
template <std::floating_point F>
class FiniteElement
{
  template <typename T, std::size_t d>
  using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

public:
  /// @brief Construct a finite element.
  ///
  /// Initialising a finite element calculates the basis functions of
  /// the finite element, in terms of the polynomial basis.
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
  /// The matrix @f$BD^{T}@f$ can be obtained from an element by using
  /// dual_matrix(). The matrix @f$C@f$ can be obtained from an element
  /// by using coefficient_matrix().
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
  /// @param[in] poly_type The polyset type
  /// @param[in] degree The degree of the element
  /// @param[in] interpolation_nderivs The number of derivatives that
  /// need to be used during interpolation
  /// @param[in] value_shape The value shape of the element
  /// @param[in] wcoeffs Matrices for the kth value index containing the
  /// expansion coefficients defining a polynomial basis spanning the
  /// polynomial space for this element. Shape is (dim(finite element
  /// polyset), dim(Legendre polynomials))
  /// @param[in] x Interpolation points. Indices are (tdim, entity
  /// index, point index, dim)
  /// @param[in] M The interpolation matrices. Indices are (tdim, entity
  /// index, dof, vs, point_index, derivative)
  /// @param[in] map_type The type of map to be used to map values from
  /// the reference to a cell
  /// @param[in] sobolev_space The underlying Sobolev space for the
  /// element
  /// @param[in] discontinuous Indicates whether or not this is the
  /// discontinuous version of the element
  /// @param[in] embedded_subdegree The highest degree n such that
  /// a Lagrange (or vector Lagrange) element of degree n is a subspace
  /// of this element
  /// @param[in] embedded_superdegree The highest degree n such that at least
  /// one polynomial of degree n is included in this element's
  /// polymonial set
  /// @param[in] lvariant The Lagrange variant of the element
  /// @param[in] dvariant The DPC variant of the element
  /// @param[in] dof_ordering DOF reordering: a mapping from the
  /// reference order to a new permuted order
  FiniteElement(element::family family, cell::type cell_type,
                polyset::type poly_type, int degree,
                const std::vector<std::size_t>& value_shape,
                mdspan_t<const F, 2> wcoeffs,
                const std::array<std::vector<mdspan_t<const F, 2>>, 4>& x,
                const std::array<std::vector<mdspan_t<const F, 4>>, 4>& M,
                int interpolation_nderivs, maps::type map_type,
                sobolev::space sobolev_space, bool discontinuous,
                int embedded_subdegree, int embedded_superdegree,
                element::lagrange_variant lvariant,
                element::dpc_variant dvariant,
                std::vector<int> dof_ordering = {});

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

  /// @brief Check if two elements are the same
  /// @note This operator compares the element properties, e.g. family,
  /// degree, etc, and not computed numerical data
  /// @return True if elements are the same
  bool operator==(const FiniteElement& e) const;

  /// Get a unique hash of this element
  std::size_t hash() const;

  /// @brief Array shape for tabulate basis values and derivatives at
  /// set of points.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] num_points Number of points that basis will be computed
  /// at.
  /// @return The shape of the array to will filled when passed to
  /// tabulate().
  std::array<std::size_t, 4> tabulate_shape(std::size_t nd,
                                            std::size_t num_points) const
  {
    std::size_t ndsize = 1;
    for (std::size_t i = 1; i <= nd; ++i)
      ndsize *= (_cell_tdim + i);
    for (std::size_t i = 1; i <= nd; ++i)
      ndsize /= i;
    std::size_t vs = std::accumulate(_value_shape.begin(), _value_shape.end(),
                                     1, std::multiplies{});
    std::size_t ndofs = _coeffs.second[0];
    return {ndsize, num_points, ndofs, vs};
  }

  /// @brief Compute basis values and derivatives at set of points.
  ///
  /// @note The version of tabulate() with the basis data as an out
  /// argument should be preferred for repeated call where performance
  /// is critical.
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
  std::pair<std::vector<F>, std::array<std::size_t, 4>>
  tabulate(int nd, impl::mdspan_t<const F, 2> x) const;

  /// @brief Compute basis values and derivatives at set of points.
  ///
  /// @note The version of tabulate() with the basis data as an out
  /// argument should be preferred for repeated call where performance
  /// is critical
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions
  /// (row-major storage).
  /// @param[in] shape The shape `(number of points, geometric
  /// dimension)` of the `x` array.
  /// @return The basis functions (and derivatives). The shape is
  /// (derivative, point, basis fn index, value index).
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, ie for
  /// the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
  /// (0,2), (3,0)... The function indexing::idx can be used to find the
  /// appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  std::pair<std::vector<F>, std::array<std::size_t, 4>>
  tabulate(int nd, std::span<const F> x,
           std::array<std::size_t, 2> shape) const;

  /// @brief Compute basis values and derivatives at set of points.
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions.
  /// The shape of x is (number of points, geometric dimension).
  /// @param [out] basis Memory location to fill. It must be allocated
  /// with shape `(num_derivatives, num_points, num basis functions,
  /// value_size)`. The function tabulate_shape() can be used to get the
  /// required shape.
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, ie for
  /// the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
  /// (0,2), (3,0)... The function indexing::idx can be used to
  /// find the appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  ///
  /// @todo Remove all internal dynamic memory allocation, pass scratch
  /// space as required
  void tabulate(int nd, impl::mdspan_t<const F, 2> x,
                mdspan_t<F, 4> basis) const;

  /// @brief Compute basis values and derivatives at set of points.
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions
  /// (row-major storage). The shape of `x` is `(number of points,
  /// geometric dimension)`.
  /// @param[in] xshape The shape `(number of points, geometric
  /// dimension)` of `x`.
  /// @param [out] basis Memory location to fill. It must be allocated
  /// with shape `(num_derivatives, num_points, num basis functions,
  /// value_size)`. The function tabulate_shape() can be used to get the
  /// required shape.
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, ie for the
  /// (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2),
  /// (3,0)... The function indexing::idx can be used to find the
  /// appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  void tabulate(int nd, std::span<const F> x, std::array<std::size_t, 2> xshape,
                std::span<F> basis) const;

  /// @brief Get the element cell type.
  /// @return The cell type
  cell::type cell_type() const { return _cell_type; }

  /// @brief Get the element polyset type.
  /// @return The polyset
  polyset::type polyset_type() const { return _poly_type; }

  /// @brief Get the element polynomial degree.
  /// @return Polynomial degree
  int degree() const { return _degree; }

  /// @brief Lowest degree `n` such that the highest degree polynomial in this
  /// element is contained in a Lagrange (or vector Lagrange) element of
  /// degree `n`.
  /// @return Polynomial degree
  int embedded_superdegree() const { return _embedded_superdegree; }

  /// @brief Highest degree `n` such that a Lagrange (or vector Lagrange)
  /// element of degree n is a subspace of this element.
  /// @return Polynomial degree
  int embedded_subdegree() const { return _embedded_subdegree; }

  /// @brief Element value tensor shape.
  ///
  /// For example, returns `{}` for scalars, `{3}` for vectors in 3D,
  /// `{2, 2}` for a rank-2 tensor in 2D.
  /// @return Value shape
  const std::vector<std::size_t>& value_shape() const { return _value_shape; }

  /// @brief Dimension of the finite element space.
  ///
  /// The dimension is the number of degrees-of-freedom for the element.
  /// @return Number of degrees of freedom
  int dim() const { return _coeffs.second[0]; }

  /// @brief The finite element family.
  /// @return The family
  element::family family() const { return _family; }

  /// @brief Lagrange variant of the element.
  /// @return The Lagrange variant.
  element::lagrange_variant lagrange_variant() const
  {
    return _lagrange_variant;
  }

  /// @brief DPC variant of the element.
  /// @return The DPC variant.
  element::dpc_variant dpc_variant() const { return _dpc_variant; }

  /// @brief Map type for the element.
  /// @return The map type.
  maps::type map_type() const { return _map_type; }

  /// @brief Underlying Sobolev space for this element.
  /// @return The Sobolev space.
  sobolev::space sobolev_space() const { return _sobolev_space; }

  /// @brief Indicates whether this element is the discontinuous
  /// variant.
  /// @return True if this element is a discontinuous version of the
  /// element.
  bool discontinuous() const { return _discontinuous; }

  /// @brief Indicates if the degree-of-freedom transformations are all
  /// permutations.
  bool dof_transformations_are_permutations() const
  {
    return _dof_transformations_are_permutations;
  }

  /// @brief Indicates is the dof transformations are all the identity.
  bool dof_transformations_are_identity() const
  {
    return _dof_transformations_are_identity;
  }

  /// @brief Map function values from the reference to a physical cell.
  ///
  /// This function can perform the mapping for multiple points, grouped
  /// by points that share a common Jacobian.
  /// @param[in] U The function values on the reference. The indices are
  /// `[Jacobian index, point index, components]`.
  /// @param[in] J The Jacobian of the mapping. The indices are
  /// `[Jacobian index, J_i, J_j]`.
  /// @param[in] detJ The determinant of the Jacobian of the mapping. It
  /// has length `J.shape(0)`
  /// @param[in] K The inverse of the Jacobian of the mapping. The
  /// indices are `[Jacobian index, K_i, K_j]`.
  /// @return The function values on the cell. The indices are [Jacobian
  /// index, point index, components].
  std::pair<std::vector<F>, std::array<std::size_t, 3>>
  push_forward(impl::mdspan_t<const F, 3> U, impl::mdspan_t<const F, 3> J,
               std::span<const F> detJ, impl::mdspan_t<const F, 3> K) const;

  /// @brief Map function values from a physical cell to the reference.
  /// @param[in] u The function values on the cell
  /// @param[in] J The Jacobian of the mapping
  /// @param[in] detJ The determinant of the Jacobian of the mapping
  /// @param[in] K The inverse of the Jacobian of the mapping
  /// @return The function values on the reference. The indices are
  /// [Jacobian index, point index, components].
  std::pair<std::vector<F>, std::array<std::size_t, 3>>
  pull_back(impl::mdspan_t<const F, 3> u, impl::mdspan_t<const F, 3> J,
            std::span<const F> detJ, impl::mdspan_t<const F, 3> K) const;

  /// @brief Return a function that performs the appropriate
  /// push-forward/pull-back for the element type.
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
  std::function<void(O&, const P&, const Q&, F, const R&)> map_fn() const
  {
    switch (_map_type)
    {
    case maps::type::identity:
      return [](O& u, const P& U, const Q&, F, const R&)
      {
        assert(U.extent(0) == u.extent(0));
        assert(U.extent(1) == u.extent(1));
        for (std::size_t i = 0; i < U.extent(0); ++i)
          for (std::size_t j = 0; j < U.extent(1); ++j)
            u(i, j) = U(i, j);
      };
    case maps::type::covariantPiola:
      return [](O& u, const P& U, const Q& J, F detJ, const R& K)
      { maps::covariant_piola(u, U, J, detJ, K); };
    case maps::type::contravariantPiola:
      return [](O& u, const P& U, const Q& J, F detJ, const R& K)
      { maps::contravariant_piola(u, U, J, detJ, K); };
    case maps::type::doubleCovariantPiola:
      return [](O& u, const P& U, const Q& J, F detJ, const R& K)
      { maps::double_covariant_piola(u, U, J, detJ, K); };
    case maps::type::doubleContravariantPiola:
      return [](O& u, const P& U, const Q& J, F detJ, const R& K)
      { maps::double_contravariant_piola(u, U, J, detJ, K); };
    default:
      throw std::runtime_error("Map not implemented");
    }
  }

  /// @brief Get the dofs on each topological entity: (vertices, edges,
  /// faces, cell) in that order.
  ///
  /// For example, Lagrange degree 2 on a triangle has vertices: [[0],
  /// [1], [2]], edges: [[3], [4], [5]], cell: [[]]
  /// @return Dofs associated with an entity of a given topological
  /// dimension. The shape is (tdim + 1, num_entities, num_dofs).
  const std::vector<std::vector<std::vector<int>>>& entity_dofs() const
  {
    return _edofs;
  }

  /// @brief Get the dofs on the closure of each topological entity:
  /// (vertices, edges, faces, cell) in that order.
  ///
  /// For example, Lagrange degree 2 on a triangle has vertices: [[0],
  /// [1], [2]], edges: [[1, 2, 3], [0, 2, 4], [0, 1, 5]], cell: [[0, 1,
  /// 2, 3, 4, 5]]
  /// @return Dofs associated with the closure of an entity of a given
  /// topological dimension. The shape is `(tdim + 1, num_entities,
  /// num_dofs)`.
  const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs() const
  {
    return _e_closure_dofs;
  }

  /// @brief Get the base transformations.
  ///
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
  ///   rotation:   [[-1, 1],
  ///                [ 1, 0]]
  ///   reflection: [[0, 1],
  ///                [1, 0]]
  /// ~~~~~~~~~~~~~~~~
  /// @return The base transformations for this element. The shape is
  /// (ntranformations, ndofs, ndofs)
  std::pair<std::vector<F>, std::array<std::size_t, 3>>
  base_transformations() const;

  /// @brief Return the entity dof transformation matrices
  /// @return The entity transformations for the sub-entities of this
  /// element. The shape for each cell is (ntransformations, ndofs,
  /// ndofs)
  std::map<cell::type, std::pair<std::vector<F>, std::array<std::size_t, 3>>>
  entity_transformations() const
  {
    return _entity_transformations;
  }

  /// @brief Permute indices associated with degree-of-freedoms on the
  /// reference element ordering to the globally consistent physical
  /// element degree-of-freedom ordering.
  ///
  /// Given an array \f$\tilde{d}\f$ that holds an integer associated
  /// with each degree-of-freedom and following the reference element
  /// degree-of-freedom ordering, this function computes
  /// \f[
  ///  d = P \tilde{d},
  /// \f]
  /// where \f$P\f$ is a permutation matrix and \f$d\f$ holds the
  /// integers in \f$\tilde{d}\f$ but permuted to follow the globally
  /// consistent physical element degree-of-freedom ordering. The
  /// permutation is computed in-place.
  ///
  /// @note This function is designed to be called at runtime, so its
  /// performance is critical.
  ///
  /// @param[in,out] d Indices associated with each reference element
  /// degree-of-freedom (in). Indices associated with each physical
  /// element degree-of-freedom (out).
  /// @param cell_info Permutation info for the cell
  void permute(std::span<std::int32_t> d, std::uint32_t cell_info) const
  {
    if (!_dof_transformations_are_permutations)
    {
      throw std::runtime_error(
          "The DOF transformations for this element are not permutations");
    }

    if (_dof_transformations_are_identity)
      return;
    else
      permute_data<std::int32_t, false>(d, 1, cell_info, _eperm);
  }

  /// @brief Perform the inverse of the operation applied by permute().
  ///
  /// Given an array \f$d\f$ that holds an integer associated with each
  /// degree-of-freedom and following the globally consistent physical
  /// element degree-of-freedom ordering, this function computes
  /// \f[
  ///  \tilde{d} = P^{T} d,
  /// \f]
  /// where \f$P^{T}\f$ is a permutation matrix and \f$\tilde{d}\f$
  /// holds the integers in \f$d\f$ but permuted to follow the reference
  /// element degree-of-freedom ordering. The permutation is computed
  /// in-place.
  ///
  /// @param[in,out] d Indices associated with each physical element
  /// degree-of-freedom [in]. Indices associated with each reference
  /// element degree-of-freedom [out].
  /// @param cell_info Permutation info for the cell
  void permute_inv(std::span<std::int32_t> d, std::uint32_t cell_info) const
  {
    if (!_dof_transformations_are_permutations)
    {
      throw std::runtime_error(
          "The DOF transformations for this element are not permutations");
    }

    if (_dof_transformations_are_identity)
      return;
    else
      permute_data<std::int32_t, true>(d, 1, cell_info, _eperm_rev);
  }

  /// @brief Transform basis functions from the reference element
  /// ordering and orientation to the globally consistent physical
  /// element ordering and orientation.
  ///
  /// Consider that the value of a finite element function \f$f_{h}\f$
  /// at a point is given by
  /// \f[
  ///  f_{h} = \phi^{T} c,
  /// \f]
  /// where \f$f_{h}\f$ has shape \f$r \times 1\f$, \f$\phi\f$ has shape
  /// \f$d \times r\f$ and holds the finite element basis functions,
  /// and \f$c\f$ has shape \f$d \times 1\f$ and holds the
  /// degrees-of-freedom. The basis functions and
  /// degree-of-freedom are with respect to the physical element
  /// orientation. If the degrees-of-freedom on the physical element
  /// orientation are given by
  ///   \f[ \phi = T \tilde{\phi}, \f]
  /// where \f$T\f$ is a \f$d \times d\f$ matrix, it follows from
  /// \f$f_{h} = \phi^{T} c = \tilde{\phi}^{T} T^{T} c\f$ that
  ///   \f[ \tilde{c} = T^{T} c. \f]
  ///
  /// This function applies \f$T\f$ to data. The transformation is
  /// performed in-place. The operator \f$T\f$ is orthogonal for many
  /// elements, but not all.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param cell_info Permutation info for the cell
  template <typename T>
  void T_apply(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Apply the transpose of the operator applied by T_apply().
  ///
  /// The transformation \f[ u \leftarrow T^{T} u \f] is performed
  /// in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom an d the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param[in] cell_info Permutation info for the cell,
  template <typename T>
  void Tt_apply(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Apply the inverse transpose of the operator applied by
  /// T_apply().
  ///
  /// The transformation \f[ u \leftarrow T^{-T} u \f] is performed
  /// in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param[in] cell_info Permutation info for the cell.
  template <typename T>
  void Tt_inv_apply(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Apply the inverse of the operator applied by T_apply().
  ///
  /// The transformation \f[ u \leftarrow T^{-1} u \f] is performed
  /// in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param[in] cell_info Permutation info for the cell.
  template <typename T>
  void Tinv_apply(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Right(post)-apply the transpose of the operator applied by
  /// T_apply().
  ///
  /// Computes \f[ u^{T} \leftarrow u^{T} T^{T} \f] in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param[in] cell_info Permutation info for the cell.
  template <typename T>
  void Tt_apply_right(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Right(post)-apply the operator applied by T_apply().
  ///
  /// Computes \f[ u^{T} \leftarrow u^{T} T \f] in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param[in] cell_info Permutation info for the cell.
  template <typename T>
  void T_apply_right(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Right(post)-apply the inverse of the operator applied by
  /// T_apply().
  ///
  /// Computes \f[ u^{T} \leftarrow u^{T} T^{-1} \f] in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param cell_info Permutation info for the cell.
  template <typename T>
  void Tinv_apply_right(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Right(post)-apply the transpose inverse of the operator
  /// applied by T_apply().
  ///
  /// Computes \f[ u^{T} \leftarrow u^{T} T^{-T} \f] in-place.
  ///
  /// @param[in,out] u Data to transform. The shape is `(m, n)`, where
  /// `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] n Number of columns in `data`.
  /// @param cell_info Permutation info for the cell.
  template <typename T>
  void Tt_inv_apply_right(std::span<T> u, int n, std::uint32_t cell_info) const;

  /// @brief Return the interpolation points.
  ///
  /// The interpolation points are the coordinates on the reference
  /// element where a function need to be evaluated in order to
  /// interpolate it in the finite element space.
  /// @return Array of coordinate with shape `(num_points, tdim)`
  const std::pair<std::vector<F>, std::array<std::size_t, 2>>& points() const
  {
    return _points;
  }

  /// @brief Return a matrix of weights interpolation.
  ///
  /// To interpolate a function in this finite element, the functions
  /// should be evaluated at each point given by points(). These
  /// function values should then be multiplied by the weight matrix to
  /// give the coefficients of the interpolated function.
  ///
  /// The shape of the returned matrix will be `(dim, num_points *
  /// value_size)`, where `dim` is the number of DOFs in the finite
  /// element, `num_points` is the number of points returned by
  /// points(), and `value_size` is the value size of the finite
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
  /// @return The interpolation matrix. Shape is `(ndofs, number of
  /// interpolation points)`.
  const std::pair<std::vector<F>, std::array<std::size_t, 2>>&
  interpolation_matrix() const
  {
    return _matM;
  }

  /// @brief Get the dual matrix.
  ///
  /// This is the matrix @f$BD^{T}@f$, as described in the documentation
  /// of the FiniteElement() constructor.
  /// @return The dual matrix. Shape is `(ndofs, ndofs)` = `(dim(), dim())`.
  const std::pair<std::vector<F>, std::array<std::size_t, 2>>&
  dual_matrix() const
  {
    return _dual_matrix;
  }

  /// @brief Get the coefficients that define the polynomial set in
  /// terms of the orthonormal polynomials.
  ///
  /// The polynomials spanned by each finite element in Basix are
  /// represented as a linear combination of the orthonormal polynomials
  /// of a given degree on the cell. Each row of this matrix defines a
  /// polynomial in the set spanned by the finite element.
  ///
  /// For example, the orthonormal polynomials of degree <= 1 on a
  /// triangle are (where a, b, c, d are some constants):
  ///
  ///  - (sqrt(2), 0)
  ///  - (a*x - b, 0)
  ///  - (c*y - d, 0)
  ///  - (0, sqrt(2))
  ///  - (0, a*x - b)
  ///  - (0, c*y - d)
  ///
  /// For a degree 1 Raviart-Thomas element, the first two rows of
  /// wcoeffs would be the following, as (1, 0) and (0, 1) are spanned
  /// by the element
  ///
  ///  - [1, 0, 0, 0, 0, 0]
  ///  - [0, 0, 0, 1, 0, 0]
  ///
  /// The third row of wcoeffs in this example would give coefficients
  /// that represent (x, y) in terms of the orthonormal polynomials:
  ///
  ///  - [-b/(a*sqrt(2)), 1/a, 0, -d/(c*sqrt(2)), 0, 1/c]
  ///
  /// These coefficients are only stored for custom elements. This
  /// function will throw an exception if called on a non-custom
  /// element.
  ///
  /// @return Coefficient matrix. Shape is `(dim(finite element polyset),
  /// dim(Lagrange polynomials))`.
  const std::pair<std::vector<F>, std::array<std::size_t, 2>>& wcoeffs() const
  {
    return _wcoeffs;
  }

  /// @brief Get the interpolation points for each subentity.
  ///
  /// The indices of this data are `(tdim, entity index, point index,
  /// dim)`.
  const std::array<
      std::vector<std::pair<std::vector<F>, std::array<std::size_t, 2>>>, 4>&
  x() const
  {
    return _x;
  }

  /// @brief Get the interpolation matrices for each subentity.
  ///
  /// The shape of this data is `(tdim, entity index, dof, value size,
  /// point_index, derivative)`.
  ///
  /// These matrices define how to evaluate the DOF functionals
  /// associated with each sub-entity of the cell. Given a function f,
  /// the functionals associated with the `e`-th entity of dimension `d`
  /// can be computed as follows:
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
  /// quadrature points for each edge, and `M()` would be a 1 by 2 by `npoints`
  /// by 1 array for each edge. For each point, the `[0, :, point, 0]` slice of
  /// this would be the quadrature weight multiplied by the normal. For all
  /// entities that are not edges, the entries in `x()` and `M()` for a degree 1
  /// RT element would have size 0.
  ///
  /// These matrices are only stored for custom elements. This function will
  /// throw an exception if called on a non-custom element
  /// @return The interpolation matrices. The indices of this data are `(tdim,
  /// entity index, dof, vs, point_index, derivative)`.
  const std::array<
      std::vector<std::pair<std::vector<F>, std::array<std::size_t, 4>>>, 4>&
  M() const
  {
    return _M;
  }

  /// @brief Get the matrix of coefficients.
  ///
  /// @return The coefficient matrix. Shape is `(ndofs, ndofs)`.
  const std::pair<std::vector<F>, std::array<std::size_t, 2>>&
  coefficient_matrix() const
  {
    return _coeffs;
  }

  /// @brief Indicates whether or not this element can be represented as a
  /// product of elements defined on lower-dimensional reference cells.
  ///
  /// If the product exists, this element's basis functions can be
  /// computed as a tensor product of the basis elements of the elements
  /// in the product.
  ///
  /// If such a factorisation exists,
  /// get_tensor_product_representation() can be used to get these
  /// elements.
  bool has_tensor_product_factorisation() const
  {
    return !_tensor_factors.empty();
  }

  /// @brief Get the tensor product representation of this element.
  ///
  /// @throws std::runtime_error Thrown if no such factorisation exists.
  ///
  /// The tensor product representation will be a vector of vectors of
  /// finite elements. Each tuple contains a vector of finite elements,
  /// and a vector of integers. The vector of finite elements gives the
  /// elements on an interval that appear in the tensor product
  /// representation. The vector of integers gives the permutation
  /// between the numbering of the tensor product DOFs and the number of
  /// the DOFs of this Basix element.
  /// @return The tensor product representation
  std::vector<std::vector<FiniteElement<F>>>
  get_tensor_product_representation() const
  {
    if (!has_tensor_product_factorisation())
      throw std::runtime_error("Element has no tensor product representation.");
    return _tensor_factors;
  }

  /// @brief Indicates whether or not the interpolation matrix for this
  /// element is an identity matrix.
  /// @return True if the interpolation matrix is the identity and false
  /// otherwise.
  bool interpolation_is_identity() const { return _interpolation_is_identity; }

  /// @brief The number of derivatives needed when interpolating
  int interpolation_nderivs() const { return _interpolation_nderivs; }

  /// @brief Get dof layout
  const std::vector<int>& dof_ordering() const { return _dof_ordering; }

private:
  /// Data permutation
  /// @param data Data to be permuted
  /// @param block_size
  /// @param cell_info Cell bitmap selecting required permutations
  /// @param eperm Permutation to use
  /// @param post Whether reflect is pre- or post- rotation.
  template <typename T, bool post>
  void permute_data(
      std::span<T> data, int block_size, std::uint32_t cell_info,
      const std::map<cell::type, std::vector<std::vector<std::size_t>>>& eperm)
      const;

  using array2_t = std::pair<std::vector<F>, std::array<std::size_t, 2>>;
  using array3_t = std::pair<std::vector<F>, std::array<std::size_t, 3>>;
  using trans_data_t
      = std::vector<std::pair<std::vector<std::size_t>, array2_t>>;

  /// Data transformation
  /// @param data Data to be transformed (using matrices)
  /// @param block_size
  /// @param cell_info Cell bitmap selecting required transforms
  /// @param etrans Transformation matrices
  /// @param post Whether reflect is pre- or post- rotation.
  template <typename T, bool post, typename OP>
  void
  transform_data(std::span<T> data, int block_size, std::uint32_t cell_info,
                 const std::map<cell::type, trans_data_t>& etrans, OP op) const;

  // Cell type
  cell::type _cell_type;

  // Polyset type
  polyset::type _poly_type;

  // Topological dimension of the cell
  std::size_t _cell_tdim;

  // Topological dimension of the cell
  std::vector<std::vector<cell::type>> _cell_subentity_types;

  // Finite element family
  element::family _family;

  // Lagrange variant
  element::lagrange_variant _lagrange_variant;

  // DPC variant
  element::dpc_variant _dpc_variant;

  // Degree that was input when creating the element
  int _degree;

  // Degree
  int _interpolation_nderivs;

  // Highest degree polynomial in element's polyset
  int _embedded_superdegree;

  // Highest degree space that is a subspace of element's polyset
  int _embedded_subdegree;

  // Value shape
  std::vector<std::size_t> _value_shape;

  /// The mapping used to map this element from the reference to a cell
  maps::type _map_type;

  /// The Sobolev space this element is contained in
  sobolev::space _sobolev_space;

  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by @f$\psi_i = \sum_{k} \phi_{k}
  // \alpha^{i}_{k}@f$, then _coeffs(i, j) = @f$\alpha^i_k@f$. ie
  // _coeffs.row(i) are the expansion coefficients for shape function i
  // (@f$\psi_{i}@f$).
  std::pair<std::vector<F>, std::array<std::size_t, 2>> _coeffs;

  // Dofs associated with each cell (sub-)entity
  std::vector<std::vector<std::vector<int>>> _edofs;

  // Dofs associated with the closdure of each cell (sub-)entity
  std::vector<std::vector<std::vector<int>>> _e_closure_dofs;

  // Entity transformations
  std::map<cell::type, array3_t> _entity_transformations;

  // Set of points used for point evaluation
  // Experimental - currently used for an implementation of
  // "tabulate_dof_coordinates" Most useful for Lagrange. This may change or go
  // away. For non-Lagrange elements, these points will be used in combination
  // with _interpolation_matrix to perform interpolation
  std::pair<std::vector<F>, std::array<std::size_t, 2>> _points;

  // Interpolation points on the cell. The shape is (entity_dim, num
  // entities of given dimension, num_points, tdim)
  std::array<std::vector<std::pair<std::vector<F>, std::array<std::size_t, 2>>>,
             4>
      _x;

  /// The interpolation weights and points
  std::pair<std::vector<F>, std::array<std::size_t, 2>> _matM;

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
  std::map<cell::type, trans_data_t> _etrans;

  // The transposed entity transformations in precomputed form
  std::map<cell::type, trans_data_t> _etransT;

  // The inverse entity transformations in precomputed form
  std::map<cell::type, trans_data_t> _etrans_inv;

  // The inverse transpose entity transformations in precomputed form
  std::map<cell::type, trans_data_t> _etrans_invT;

  // Indicates whether or not this is the discontinuous version of the
  // element
  bool _discontinuous;

  // The dual matrix
  std::pair<std::vector<F>, std::array<std::size_t, 2>> _dual_matrix;

  // Dof reordering for different element dof layout compatibility.
  // The reference basix layout is ordered by entity, i.e. dofs on
  // vertices, followed by edges, faces, then internal dofs.
  // _dof_ordering stores the map to the new order required, e.g.
  // for a P2 triangle, _dof_ordering=[0 3 5 1 2 4] will place
  // dofs 0, 3, 5 on the vertices and 1, 2, 4, on the edges.
  std::vector<int> _dof_ordering;

  // Tensor product representation
  // Entries of tuple are (list of elements on an interval, permutation
  // of DOF numbers)
  // @todo: For vector-valued elements, a tensor product type and a
  // scaling factor may additionally be needed.
  std::vector<std::vector<FiniteElement>> _tensor_factors;

  // Is the interpolation matrix an identity?
  bool _interpolation_is_identity;

  // The coefficients that define the polynomial set in terms of the
  // orthonormal polynomials
  std::pair<std::vector<F>, std::array<std::size_t, 2>> _wcoeffs;

  // Interpolation matrices for each entity
  using array4_t
      = std::vector<std::pair<std::vector<F>, std::array<std::size_t, 4>>>;
  std::array<array4_t, 4> _M;
};

/// Create a custom finite element
/// @param[in] cell_type The cell type
/// @param[in] value_shape The value shape of the element
/// @param[in] wcoeffs Matrices for the kth value index containing the
/// expansion coefficients defining a polynomial basis spanning the
/// polynomial space for this element. Shape is (dim(finite element polyset),
/// dim(Legendre polynomials))
/// @param[in] x Interpolation points. Indices are (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index, derivative)
/// @param[in] interpolation_nderivs The number of derivatives that need to be
/// used during interpolation
/// @param[in] map_type The type of map to be used to map values from
/// the reference to a cell
/// @param[in] sobolev_space The underlying Sobolev space for the element
/// @param[in] discontinuous Indicates whether or not this is the
/// discontinuous version of the element
/// @param[in] embedded_subdegree The highest degree n such that a
/// Lagrange (or vector Lagrange) element of degree n is a subspace of this
/// element
/// @param[in] embedded_superdegree The degree of a polynomial in this element's
/// polyset
/// @param[in] poly_type The type of polyset to use for this element
/// @return A custom finite element
template <std::floating_point T>
FiniteElement<T> create_custom_element(
    cell::type cell_type, const std::vector<std::size_t>& value_shape,
    impl::mdspan_t<const T, 2> wcoeffs,
    const std::array<std::vector<impl::mdspan_t<const T, 2>>, 4>& x,
    const std::array<std::vector<impl::mdspan_t<const T, 4>>, 4>& M,
    int interpolation_nderivs, maps::type map_type,
    sobolev::space sobolev_space, bool discontinuous, int embedded_subdegree,
    int embedded_superdegree, polyset::type poly_type);

/// Create an element using a given Lagrange variant and a given DPC variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @param[in] dof_ordering Ordering of dofs for ElementDofLayout
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_element(element::family family, cell::type cell,
                                int degree, element::lagrange_variant lvariant,
                                element::dpc_variant dvariant,
                                bool discontinuous,
                                std::vector<int> dof_ordering = {});

/// Get the tensor product DOF ordering for an element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on.
/// Currently limited to quadrilateral or hexahedron.
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A vector containing the dof ordering
std::vector<int> tp_dof_ordering(element::family family, cell::type cell,
                                 int degree, element::lagrange_variant lvariant,
                                 element::dpc_variant dvariant,
                                 bool discontinuous);

/// Get the tensor factors of an element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on.
/// Currently limited to quadrilateral or hexahedron.
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @param[in] dof_ordering The ordering of the DOFs
/// @return A list of lists of finite element factors
template <std::floating_point T>
std::vector<std::vector<FiniteElement<T>>>
tp_factors(element::family family, cell::type cell, int degree,
           element::lagrange_variant lvariant, element::dpc_variant dvariant,
           bool discontinuous, std::vector<int> dof_ordering);

/// Create an element with Tensor Product dof ordering
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on.
/// Currently limited to quadrilateral or hexahedron.
/// @param[in] degree The degree of the element
/// @param[in] lvariant The variant of Lagrange to use
/// @param[in] dvariant The variant of DPC to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
/// @return A finite element
template <std::floating_point T>
FiniteElement<T>
create_tp_element(element::family family, cell::type cell, int degree,
                  element::lagrange_variant lvariant,
                  element::dpc_variant dvariant, bool discontinuous);

/// Return the Basix version number
/// @return version string
std::string version();

//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T, bool post>
void FiniteElement<F>::permute_data(
    std::span<T> data, int block_size, std::uint32_t cell_info,
    const std::map<cell::type, std::vector<std::vector<std::size_t>>>& eperm)
    const
{
  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if 3D
    // cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _edofs[2].size() : 0;

    // Permute DOFs on edges
    {
      auto& trans = eperm.at(cell::type::interval)[0];
      for (std::size_t e = 0; e < _edofs[1].size(); ++e)
      {
        // Reverse an edge
        if (cell_info >> (face_start + e) & 1)
        {
          precompute::apply_permutation_mapped(trans, data, _edofs[1][e],
                                               block_size);
        }
      }
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        auto& trans = eperm.at(_cell_subentity_types[2][f]);

        // Reflect a face (pre rotate)
        if (!post and cell_info >> (3 * f) & 1)
        {
          precompute::apply_permutation_mapped(trans[1], data, _edofs[2][f],
                                               block_size);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          precompute::apply_permutation_mapped(trans[0], data, _edofs[2][f],
                                               block_size);
        }

        // Reflect a face (post rotate)
        if (post and cell_info >> (3 * f) & 1)
        {
          precompute::apply_permutation_mapped(trans[1], data, _edofs[2][f],
                                               block_size);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T, bool post, typename OP>
void FiniteElement<F>::transform_data(
    std::span<T> data, int block_size, std::uint32_t cell_info,
    const std::map<cell::type, trans_data_t>& etrans, OP op) const
{
  if (_cell_tdim >= 2)
  {
    // This assumes 3 bits are used per face. This will need updating if
    // 3D cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _edofs[2].size() : 0;
    int dofstart = 0;
    for (auto& edofs0 : _edofs[0])
      dofstart += edofs0.size();

    // Transform DOFs on edges
    {
      auto& [v_size_t, matrix] = etrans.at(cell::type::interval)[0];
      for (std::size_t e = 0; e < _edofs[1].size(); ++e)
      {
        // Reverse an edge
        if (cell_info >> (face_start + e) & 1)
        {
          op(std::span(v_size_t),
             mdspan_t<const F, 2>(matrix.first.data(), matrix.second), data,
             dofstart, block_size);
        }
        dofstart += _edofs[1][e].size();
      }
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _edofs[2].size(); ++f)
      {
        auto& trans = etrans.at(_cell_subentity_types[2][f]);

        // Reflect a face (pre rotation)
        if (!post and cell_info >> (3 * f) & 1)
        {
          const auto& m = trans[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& matrix = std::get<1>(m);
          op(std::span(v_size_t),
             mdspan_t<const F, 2>(matrix.first.data(), matrix.second), data,
             dofstart, block_size);
        }

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
        {
          const auto& m = trans[0];
          const auto& v_size_t = std::get<0>(m);
          const auto& matrix = std::get<1>(m);
          op(std::span(v_size_t),
             mdspan_t<const F, 2>(matrix.first.data(), matrix.second), data,
             dofstart, block_size);
        }

        // Reflect a face (post rotation)
        if (post and cell_info >> (3 * f) & 1)
        {
          const auto& m = trans[1];
          const auto& v_size_t = std::get<0>(m);
          const auto& matrix = std::get<1>(m);
          op(std::span(v_size_t),
             mdspan_t<const F, 2>(matrix.first.data(), matrix.second), data,
             dofstart, block_size);
        }

        dofstart += _edofs[2][f].size();
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::T_apply(std::span<T> u, int n,
                               std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;

  if (_dof_transformations_are_permutations)
    permute_data<T, false>(u, n, cell_info, _eperm);
  else
  {
    transform_data<T, false>(u, n, cell_info, _etrans,
                             precompute::apply_matrix<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::Tt_apply(std::span<T> u, int n,
                                std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
    permute_data<T, true>(u, n, cell_info, _eperm_rev);
  else
  {
    transform_data<T, true>(u, n, cell_info, _etransT,
                            precompute::apply_matrix<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::Tt_inv_apply(std::span<T> u, int n,
                                    std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
    permute_data<T, false>(u, n, cell_info, _eperm);
  else
  {
    transform_data<T, false>(u, n, cell_info, _etrans_invT,
                             precompute::apply_matrix<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::Tinv_apply(std::span<T> u, int n,
                                  std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
    permute_data<T, true>(u, n, cell_info, _eperm_rev);
  else
  {
    transform_data<T, true>(u, n, cell_info, _etrans_inv,
                            precompute::apply_matrix<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::Tt_apply_right(std::span<T> u, int n,
                                      std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
  {
    assert(u.size() % n == 0);
    const int step = u.size() / n;
    for (int i = 0; i < n; ++i)
    {
      std::span<T> dblock(u.data() + i * step, step);
      permute_data<T, false>(dblock, 1, cell_info, _eperm);
    }
  }
  else
  {
    transform_data<T, false>(u, n, cell_info, _etrans,
                             precompute::apply_tranpose_matrix_right<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::Tinv_apply_right(std::span<T> u, int n,
                                        std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
  {
    assert(u.size() % n == 0);
    const int step = u.size() / n;
    for (int i = 0; i < n; ++i)
    {
      std::span<T> dblock(u.data() + i * step, step);
      permute_data<T, false>(dblock, 1, cell_info, _eperm);
    }
  }
  else
  {
    transform_data<T, false>(u, n, cell_info, _etrans_invT,
                             precompute::apply_tranpose_matrix_right<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::T_apply_right(std::span<T> u, int n,
                                     std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
  {
    assert(u.size() % n == 0);
    const int step = u.size() / n;
    for (int i = 0; i < n; ++i)
    {
      std::span<T> dblock(u.data() + i * step, step);
      permute_data<T, true>(dblock, 1, cell_info, _eperm_rev);
    }
  }
  else
  {
    transform_data<T, true>(u, n, cell_info, _etransT,
                            precompute::apply_tranpose_matrix_right<F, T>);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point F>
template <typename T>
void FiniteElement<F>::Tt_inv_apply_right(std::span<T> u, int n,
                                          std::uint32_t cell_info) const
{
  if (_dof_transformations_are_identity)
    return;
  else if (_dof_transformations_are_permutations)
  {
    assert(u.size() % n == 0);
    const int step = u.size() / n;
    for (int i = 0; i < n; ++i)
    {
      std::span<T> dblock(u.data() + i * step, step);
      permute_data<T, true>(dblock, 1, cell_info, _eperm_rev);
    }
  }
  else
  {
    transform_data<T, true>(u, n, cell_info, _etrans_inv,
                            precompute::apply_tranpose_matrix_right<F, T>);
  }
}
//-----------------------------------------------------------------------------

} // namespace basix
