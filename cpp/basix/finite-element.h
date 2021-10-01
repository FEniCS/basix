// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

// FIXME: just include everything for now
// Need to define public API

#pragma once

#include "cell.h"
#include "element-families.h"
#include "maps.h"
#include "precompute.h"
#include <array>
#include <numeric>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

/// Placeholder
namespace basix
{

/// Calculates the basis functions of the finite element, in terms of the
/// polynomial basis.
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
/// @f$\{f_i\}@f$. The basis functions are the functions in span{@f$q_k@f$} such
/// that
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
/// This function takes the matrices B (span_coeffs) and D (dual) as
/// inputs and returns the matrix C.
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
/// @param[in] cell_type The cells shape
/// @param[in] B Matrices for the kth value index containing the
/// expansion coefficients defining a polynomial basis spanning the
/// polynomial space for this element
/// @param[in] M The interpolation tensor, such that the dual matrix
/// \f$D\f$ is computed by \f$D = MP\f$
/// @param[in] x The interpolation points. The vector index is for
/// points on entities of the same dimension, ordered with the lowest
/// topological dimension being first. Each 3D tensor hold the points on
/// cell entities of a common dimension. The shape of the 3d tensors is
/// (num_entities, num_points_per_entity, tdim).
/// @param[in] degree The degree of the polynomial basis P used to
/// create the element (before applying B)
/// @return The matrix C of expansion coefficients that define the basis
/// functions of the finite element space. The shape is (num_dofs,
/// value_size, basis_dim)
xt::xtensor<double, 3> compute_expansion_coefficients(
    cell::type cell_type, const xt::xtensor<double, 2>& B,
    const std::vector<std::vector<xt::xtensor<double, 3>>>& M,
    const std::vector<std::vector<xt::xtensor<double, 2>>>& x, int degree);

/// Creates a version of the interpolation points, interpolation matrices and
/// entity transformation that represent a discontinuous version of the element.
/// This discontinuous version will have the same DOFs but they will all be
/// associated with the interior of the reference cell.
/// @param[in] x Interpolation points. Shape is (tdim, entity index,
/// point index, dim)
/// @param[in] M The interpolation matrices. Indices are (tdim, entity
/// index, dof, vs, point_index)
/// @param[in] entity_transformations Entity transformations
/// @param[in] tdim The topological dimension of the cell the element is defined
/// on
/// @param[in] value_size The value size of the element
std::tuple<std::array<std::vector<xt::xtensor<double, 2>>, 4>,
           std::array<std::vector<xt::xtensor<double, 3>>, 4>,
           std::map<cell::type, xt::xtensor<double, 3>>>
make_discontinuous(
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    std::map<cell::type, xt::xtensor<double, 3>>& entity_transformations,
    const int tdim, const int value_size);

/// Finite Element
/// The basis is stored as a set of coefficients, which are applied to the
/// underlying expansion set for that cell type, when tabulating.
class FiniteElement
{

public:
  /// @todo Document
  /// A finite element
  /// @param[in] family
  /// @param[in] cell_type
  /// @param[in] degree
  /// @param[in] value_shape
  /// @param[in] coeffs Expansion coefficients. The shape is (num_dofs,
  /// value_size, basis_dim)
  /// @param[in] entity_transformations Entity transformations
  /// @param[in] x Interpolation points. Shape is (tdim, entity index,
  /// point index, dim)
  /// @param[in] M The interpolation matrices. Indices are (tdim, entity
  /// index, dof, vs, point_index)
  /// @param[in] map_type The type of map to be used to map values from the
  /// reference to a cell
  /// @param[in] discontinuous Indicates whether or not this is the
  /// discontinuous version of the element
  FiniteElement(element::family family, cell::type cell_type, int degree,
                const std::vector<std::size_t>& value_shape,
                const xt::xtensor<double, 3>& coeffs,
                const std::map<cell::type, xt::xtensor<double, 3>>&
                    entity_transformations,
                const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
                const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
                maps::type map_type, bool discontinuous);

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

  /// Compute basis values and derivatives at set of points.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions.
  /// The shape of x is (number of points, geometric dimension).
  /// @return The basis functions (and derivatives). The shape is
  /// (derivative, point, basis fn index, value index).
  /// - The first index is the derivative, with higher derivatives are
  /// stored in triangular (2D) or tetrahedral (3D) ordering, i.e. for
  /// the (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1),
  /// (0,2), (3,0)... The function basix::idx can be used to find the
  /// appropriate derivative.
  /// - The second index is the point index
  /// - The third index is the basis function index
  /// - The fourth index is the basis function component. Its has size
  /// one for scalar basis functions.
  xt::xtensor<double, 4> tabulate(int nd, const xt::xarray<double>& x) const;

  /// Direct to memory block tabulation
  /// @param nd Number of derivatives
  /// @param x Points
  /// @param basis_data Memory location to fill
  void tabulate(int nd, const xt::xarray<double>& x,
                xt::xtensor<double, 4>& basis_data) const;

  /// Get the element cell type
  /// @return The cell type
  cell::type cell_type() const;

  /// Get the element polynomial degree
  /// @return Polynomial degree
  int degree() const;

  /// Get the element value size
  /// This is just a convenience function returning product(value_shape)
  /// @return Value size
  int value_size() const;

  /// Get the element value tensor shape, e.g. returning [1] for scalars.
  /// @return Value shape
  const std::vector<int>& value_shape() const;

  /// Dimension of the finite element space (number of degrees of
  /// freedom for the element)
  /// @return Number of degrees of freedom
  int dim() const;

  /// Get the finite element family
  /// @return The family
  element::family family() const;

  /// Get the mapping type used for this element
  /// @return The mapping
  maps::type mapping_type() const;

  /// Indicates whether this element is the discontinuous variant
  /// @return True if this element is a discontinuous version
  /// of the element
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
  ///
  /// @param U The function values on the reference. The indices are
  /// [Jacobian index, point index, components].
  /// @param J The Jacobian of the mapping. The indices are [Jacobian
  /// index, J_i, J_j].
  /// @param detJ The determinant of the Jacobian of the mapping. It has
  /// length `J.shape(0)`
  /// @param K The inverse of the Jacobian of the mapping. The indices
  /// are [Jacobian index, K_i, K_j].
  /// @return The function values on the cell. The indices are [Jacobian
  /// index, point index, components].
  xt::xtensor<double, 3>
  map_push_forward(const xt::xtensor<double, 3>& U,
                   const xt::xtensor<double, 3>& J,
                   const xtl::span<const double>& detJ,
                   const xt::xtensor<double, 3>& K) const;

  /// Direct to memory push forward
  ///
  /// @param[in] U Data defined on the reference element. It must have
  /// dimension 3. The first index is for the geometric/map data, the
  /// second is the point index for points that share map data, and the
  /// third index is (vector) component, e.g. `u[i,:,:]` are points that
  /// are mapped by `J[i,:,:]`.
  /// @param[in] J The Jacobians. It must have dimension 3. The first
  /// index is for the ith Jacobian, i.e. J[i,:,:] is the ith Jacobian.
  /// @param[in] detJ The determinant of J. `detJ[i]` is equal to
  /// `det(J[i,:,:])`. It must have dimension 1. @param[in] K The
  /// inverse of J, `K[i,:,:] = J[i,:,:]^-1`. It must
  /// have dimension 3.
  /// @param[out] u The input `U` mapped to the physical. It must have
  /// dimension 3.
  template <typename O, typename P, typename Q, typename S, typename T>
  void map_push_forward_m(const O& U, const P& J, const Q& detJ, const S& K,
                          T&& u) const
  {
    // FIXME: Should U.shape(2) be replaced by the physical value size?
    // Can it differ?

    // Loop over points that share J
    for (std::size_t i = 0; i < U.shape(0); ++i)
    {
      auto _J = xt::view(J, i, xt::all(), xt::all());
      auto _K = xt::view(K, i, xt::all(), xt::all());
      auto _U = xt::view(U, i, xt::all(), xt::all());
      auto _u = xt::view(u, i, xt::all(), xt::all());
      maps::apply_map(_u, _U, _J, detJ[i], _K, map_type);
    }
  }

  /// Map function values from a physical cell to the reference
  /// @param[in] u The function values on the cell
  /// @param[in] J The Jacobian of the mapping
  /// @param[in] detJ The determinant of the Jacobian of the mapping
  /// @param[in] K The inverse of the Jacobian of the mapping
  /// @return The function values on the reference
  xt::xtensor<double, 3> map_pull_back(const xt::xtensor<double, 3>& u,
                                       const xt::xtensor<double, 3>& J,
                                       const xtl::span<const double>& detJ,
                                       const xt::xtensor<double, 3>& K) const;

  /// Map function values from a physical cell back to to the reference
  ///
  /// @param[in] u Data defined on the physical element. It must have
  /// dimension 3. The first index is for the geometric/map data, the
  /// second is the point index for points that share map data, and the
  /// third index is (vector) component, e.g. `u[i,:,:]` are points that
  /// are mapped by `J[i,:,:]`.
  /// @param[in] J The Jacobians. It must have dimension 3. The first
  /// index is for the ith Jacobian, i.e. J[i,:,:] is the ith Jacobian.
  /// @param[in] detJ The determinant of J. `detJ[i]` is equal to
  /// `det(J[i,:,:])`. It must have dimension 1. @param[in] K The
  /// inverse of J, `K[i,:,:] = J[i,:,:]^-1`. It must
  /// have dimension 3.
  /// @param[out] U The input `u` mapped to the reference element. It
  /// must have dimension 3.
  template <typename O, typename P, typename Q, typename S, typename T>
  void map_pull_back_m(const O& u, const P& J, const Q& detJ, const S& K,
                       T&& U) const
  {
    // Loop over points that share K and K
    for (std::size_t i = 0; i < u.shape(0); ++i)
    {
      auto _J = xt::view(J, i, xt::all(), xt::all());
      auto _K = xt::view(K, i, xt::all(), xt::all());
      auto _u = xt::view(u, i, xt::all(), xt::all());
      auto _U = xt::view(U, i, xt::all(), xt::all());
      maps::apply_map(_U, _u, _K, 1.0 / detJ[i], _J, map_type);
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
  xt::xtensor<double, 3> base_transformations() const;

  /// Return the entity dof transformation matricess
  std::map<cell::type, xt::xtensor<double, 3>> entity_transformations() const;

  /// Permute the dof numbering on a cell
  /// @param[in,out] dofs The dof numbering for the cell
  /// @param cell_info The permutation info for the cell
  void permute_dofs(const xtl::span<std::int32_t>& dofs,
                    std::uint32_t cell_info) const;

  /// Unpermute the dof numbering on a cell
  /// @param[in,out] dofs The dof numbering for the cell
  /// @param cell_info The permutation info for the cell
  void unpermute_dofs(const xtl::span<std::int32_t>& dofs,
                      std::uint32_t cell_info) const;

  /// Apply DOF transformations to some data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_dof_transformation(const xtl::span<T>& data, int block_size,
                                std::uint32_t cell_info) const;

  /// Apply transpose DOF transformations to some data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_transpose_dof_transformation(const xtl::span<T>& data,
                                          int block_size,
                                          std::uint32_t cell_info) const;

  /// Apply inverse transpose DOF transformations to some data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_transpose_dof_transformation(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Apply inverse DOF transformations to some data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_dof_transformation(const xtl::span<T>& data,
                                        int block_size,
                                        std::uint32_t cell_info) const;

  /// Apply DOF transformations to some transposed data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_dof_transformation_to_transpose(const xtl::span<T>& data,
                                             int block_size,
                                             std::uint32_t cell_info) const;

  /// Apply transpose DOF transformations to some transposed data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_transpose_dof_transformation_to_transpose(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Apply inverse transpose DOF transformations to some transposed data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_transpose_dof_transformation_to_transpose(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Apply inverse DOF transformations to some transposed data
  /// @param[in,out] data The data
  /// @param block_size The number of data points per DOF
  /// @param cell_info The permutation info for the cell
  template <typename T>
  void apply_inverse_dof_transformation_to_transpose(
      const xtl::span<T>& data, int block_size, std::uint32_t cell_info) const;

  /// Return the interpolation points, i.e. the coordinates on the
  /// reference element where a function need to be evaluated in order
  /// to interpolate it in the finite element space.
  /// @return Array of coordinate with shape `(num_points, tdim)`
  const xt::xtensor<double, 2>& points() const;

  /// Return the number of interpolation points
  int num_points() const;

  /// Return a matrix of weights interpolation
  /// To interpolate a function in this finite element, the functions
  /// should be evaluated at each point given by
  /// FiniteElement::points(). These function values should then be
  /// multiplied by the weight matrix to give the coefficients of the
  /// interpolated function.
  const xt::xtensor<double, 2>& interpolation_matrix() const;

  /// Compute the coefficients of a function given the values of the function
  /// at the interpolation points.
  /// @param[in,out] coefficients The coefficients of the function's
  /// interpolation into the function space
  /// @param[in] data The function evaluated at the points given by `points()`
  /// @param[in] block_size The block size of the data
  template <typename T>
  void interpolate(const xtl::span<T>& coefficients,
                   const xtl::span<const T>& data, const int block_size) const;

  /// Element map type
  maps::type map_type;

private:
  // Cell type
  cell::type _cell_type;

  // Topological dimension of the cell
  std::size_t _cell_tdim;

  // Topological dimension of the cell
  std::vector<std::vector<cell::type>> _cell_subentity_types;

  // Finite element family
  element::family _family;

  // Degree
  int _degree;

  // Value shape
  std::vector<int> _value_shape;

  /// The mapping used to map this element from the reference to a cell
  maps::type _map_type;

  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by @f$\psi_i = \sum_{k} \phi_{k}
  // \alpha^{i}_{k}@f$, then _coeffs(i, j) = @f$\alpha^i_k@f$. i.e.,
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

  // Number of dofs associated with the closure of each cell (sub-)entity
  std::vector<std::vector<int>> _num_e_closure_dofs;

  // Dofs associated with each cell (sub-)entity
  std::vector<std::vector<std::vector<int>>> _edofs;

  // Dofs associated with each cell (sub-)entity
  std::vector<std::vector<std::vector<int>>> _e_closure_dofs;

  // Entity transformations
  std::map<cell::type, xt::xtensor<double, 3>> _entity_transformations;

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

  /// Interpolation matrices
  std::array<std::vector<xt::xtensor<double, 3>>, 4> _matM_new;

  /// Indicates whether or not the DOF transformations are all permutations
  bool _dof_transformations_are_permutations;

  /// Indicates whether or not the DOF transformations are all identity
  bool _dof_transformations_are_identity;

  /// The entity permutations (factorised). This will only be set if
  /// _dof_transformations_are_permutations is True and
  /// _dof_transformations_are_identity is False
  std::map<cell::type, std::vector<std::vector<std::size_t>>> _eperm;

  /// The reverse entity permutations (factorised). This will only be set if
  /// _dof_transformations_are_permutations is True and
  /// _dof_transformations_are_identity is False
  std::map<cell::type, std::vector<std::vector<std::size_t>>> _eperm_rev;

  /// The entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etrans;

  /// The transposed entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etransT;

  /// The inverse entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etrans_inv;

  /// The inverse transpose entity transformations in precomputed form
  std::map<cell::type,
           std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>,
                                  xt::xtensor<double, 2>>>>
      _etrans_invT;

  // Indicates whether or not this is the discontinuous version of the element
  bool _discontinuous;
};

/// Create an element using a given Lagrange variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of Lagrange to use
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::lagrange_variant variant,
                             bool discontinuous);

/// Create an element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Indicates whether the element is discontinuous
/// between cells points of the element. The discontinuous element will have the
/// same DOFs, but they will all be associated with the interior of the cell.
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, bool discontinuous);

/// Create a continuous element using a given Lagrange variant
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of Lagrange to use
FiniteElement create_element(element::family family, cell::type cell,
                             int degree, element::lagrange_variant variant);

/// Create a continuous element
/// @param[in] family The element family
/// @param[in] cell The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
FiniteElement create_element(element::family family, cell::type cell,
                             int degree);

/// Return the version number of basix across projects
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
        precompute::apply_matrix(_etrans.at(cell::type::interval)[0], data,
                                 dofstart, block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix(_etrans.at(_cell_subentity_types[2][f])[1],
                                   data, dofstart, block_size);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix(_etrans.at(_cell_subentity_types[2][f])[0],
                                   data, dofstart, block_size);
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
        precompute::apply_matrix(_etransT.at(cell::type::interval)[0], data,
                                 dofstart, block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix(_etransT.at(_cell_subentity_types[2][f])[0],
                                   data, dofstart, block_size);
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix(_etransT.at(_cell_subentity_types[2][f])[1],
                                   data, dofstart, block_size);
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
    // This assumes 3 bits are used per face. This will need updating if 3D
    // cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
        precompute::apply_matrix(_etrans_invT.at(cell::type::interval)[0], data,
                                 dofstart, block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix(
              _etrans_invT.at(_cell_subentity_types[2][f])[1], data, dofstart,
              block_size);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix(
              _etrans_invT.at(_cell_subentity_types[2][f])[0], data, dofstart,
              block_size);
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
    // This assumes 3 bits are used per face. This will need updating if 3D
    // cells with faces with more than 4 sides are implemented
    int face_start = _cell_tdim == 3 ? 3 * _num_edofs[2].size() : 0;
    int dofstart
        = std::accumulate(_num_edofs[0].cbegin(), _num_edofs[0].cend(), 0);

    // Transform DOFs on edges
    for (std::size_t e = 0; e < _num_edofs[1].size(); ++e)
    {
      // Reverse an edge
      if (cell_info >> (face_start + e) & 1)
        precompute::apply_matrix(_etrans_inv.at(cell::type::interval)[0], data,
                                 dofstart, block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix(
              _etrans_inv.at(_cell_subentity_types[2][f])[0], data, dofstart,
              block_size);
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix(
              _etrans_inv.at(_cell_subentity_types[2][f])[1], data, dofstart,
              block_size);
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
        precompute::apply_matrix_to_transpose(
            _etrans.at(cell::type::interval)[0], data, dofstart, block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix_to_transpose(
              _etrans.at(_cell_subentity_types[2][f])[1], data, dofstart,
              block_size);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix_to_transpose(
              _etrans.at(_cell_subentity_types[2][f])[0], data, dofstart,
              block_size);
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
        precompute::apply_matrix_to_transpose(
            _etrans_invT.at(cell::type::interval)[0], data, dofstart,
            block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix_to_transpose(
              _etrans_invT.at(_cell_subentity_types[2][f])[1], data, dofstart,
              block_size);

        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix_to_transpose(
              _etrans_invT.at(_cell_subentity_types[2][f])[0], data, dofstart,
              block_size);
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
        precompute::apply_matrix_to_transpose(
            _etransT.at(cell::type::interval)[0], data, dofstart, block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix_to_transpose(
              _etransT.at(_cell_subentity_types[2][f])[0], data, dofstart,
              block_size);

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix_to_transpose(
              _etransT.at(_cell_subentity_types[2][f])[1], data, dofstart,
              block_size);

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
        precompute::apply_matrix_to_transpose(
            _etrans_inv.at(cell::type::interval)[0], data, dofstart,
            block_size);
      dofstart += _num_edofs[1][e];
    }

    if (_cell_tdim == 3)
    {
      // Permute DOFs on faces
      for (std::size_t f = 0; f < _num_edofs[2].size(); ++f)
      {
        // Rotate a face
        for (std::uint32_t r = 0; r < (cell_info >> (3 * f + 1) & 3); ++r)
          precompute::apply_matrix_to_transpose(
              _etrans_inv.at(_cell_subentity_types[2][f])[0], data, dofstart,
              block_size);

        // Reflect a face
        if (cell_info >> (3 * f) & 1)
          precompute::apply_matrix_to_transpose(
              _etrans_inv.at(_cell_subentity_types[2][f])[1], data, dofstart,
              block_size);

        dofstart += _num_edofs[2][f];
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::interpolate(const xtl::span<T>& coefficients,
                                const xtl::span<const T>& data,
                                const int block_size) const
{
  if (block_size != 1)
  {
    throw std::runtime_error(
        "Interpolation of blocked data not implemented yet.");
  }

  const std::size_t rows = dim();

  // Compute coefficients = Pi * x (matrix-vector multiply)
  const xt::xtensor<double, 2>& Pi = interpolation_matrix();
  assert(Pi.size() % rows == 0);
  const std::size_t cols = Pi.size() / rows;
  for (std::size_t i = 0; i < rows; ++i)
  {
    // Can be replaced with std::transform_reduce once GCC 8 series dies.
    // Dot product between row i of the matrix and 'data'
    coefficients[i] = std::inner_product(std::next(Pi.data(), i * cols),
                                         std::next(Pi.data(), i * cols + cols),
                                         data.data(), T(0.0));
  }
}
//-----------------------------------------------------------------------------

} // namespace basix
