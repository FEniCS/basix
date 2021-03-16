// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

// FIXME: just include everything for now
// Need to define public API

#pragma once

#include "cell.h"
#include "element-families.h"
#include "mappings.h"
#include "span.hpp"
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

/// Placeholder
namespace basix
{

/// Calculates the basis functions of the finite element, in terms of the
/// polynomial basis.
///
/// The basis functions @f$(\phi_i)@f$ of a finite element can be represented
/// as a linear combination of polynomials @f$(p_j)@f$ in an underlying
/// polynomial basis that span the space of all d-dimensional polynomials up
/// to order
/// @f$k (P_k^d)@f$:
/// @f[  \phi_i = \sum_j c_{ij} p_j @f]
/// This function computed the matrix @f$C = (c_{ij})@f$.
///
/// In some cases, the basis functions @f$(\phi_i)@f$ do not span the full
/// space
/// @f$P_k@f$. In these cases, we represent the space spanned by the basis
/// functions as the span of some polynomials @f$(q_k)@f$. These can be
/// represented in terms of the underlying polynomial basis:
/// @f[  q_k = \sum_j b_{kj} p_j @f]
/// If the basis functions span the full space, then @f$B = (b_{kj})@f$ is
/// simply the identity.
///
/// The basis functions @f$\phi_i@f$ are defined by a dual set of functionals
/// @f$(f_l)@f$. The basis functions are the functions in span{@f$q_k@f$} such
/// that:
///   @f[ f_l(\phi_i) = 1 \mbox{ if } i=l \mbox{ else } 0 @f]
/// We can define a matrix D given by applying the functionals to each
/// polynomial p_j:
///  @f[ D = (d_{lj}),\mbox{ where } d_{lj} = f_l(p_j) @f]
///
/// This function takes the matrices B (span_coeffs) and D (dual) as
/// inputs and returns the matrix C. It computed C using:
///  @f[ C = (B D^T)^{-1} B @f]
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
///   dual = @f[ \begin{bmatrix}
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
/// @param[in] B The matrix containing the expansion coefficients
/// defining a polynomial basis spanning the polynomial space for this
/// element
/// @param[in] M The interpolation matrix, such that the dual matrix
/// \f$D\f$ is computed by \f$D = MP\f$
/// @param[in] x The interpolation points
/// @param[in] degree The degree of the polynomial set
/// @param[in] kappa_tol If positive, the condition number is computed
/// and an error thrown if the condition number of \f$B D^{T}\f$ is
/// greater than @p kappa_tol. If @p kappa_tol is less than 1 the
/// condition number is not checked.
/// @return The matrix C of expansion coefficients that define the basis
/// functions of the finite element space.
xt::xtensor<double, 2> compute_expansion_coefficients(
    cell::type cell_type, const xt::xtensor<double, 2>& B,
    const xt::xtensor<double, 2>& M, const xt::xtensor<double, 2>& x,
    int degree, double kappa_tol = 0.0);

/// Combines interpolation data
///
/// When the value size is not 1, the matrices are split up into
/// `value_size` parts, then recombined so that the columns of the
/// matrix that is output is ordered correctly.
///
/// @param[in] points_1d The interpolation points for a 1d entity
/// @param[in] points_2d The interpolation points for a 2d entity
/// @param[in] points_3d The interpolation points for a 3d entity
/// @param[in] matrix_1d The interpolation matrix for a 1d entity
/// @param[in] matrix_2d The interpolation matrix for a 2d entity
/// @param[in] matrix_3d The interpolation matrix for a 3d entity
/// @param[in] tdim The toplogical dimension
/// @param[in] value_size Value size
/// @return The interpolation points and matrix
std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 2>>
combine_interpolation_data(const xt::xtensor<double, 2>& points_1d,
                           const xt::xtensor<double, 2>& points_2d,
                           const xt::xtensor<double, 2>& points_3d,
                           const xt::xtensor<double, 2>& matrix_1d,
                           const xt::xtensor<double, 2>& matrix_2d,
                           const xt::xtensor<double, 2>& matrix_3d,
                           std::size_t tdim, std::size_t value_size);

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
  /// @param[in] coeffs
  /// @param[in] entity_dofs
  /// @param[in] base_transformations Base transformations
  /// @param[in] points
  /// @param[in] M The interpolation matrix
  /// @param[in] map_type
  FiniteElement(element::family family, cell::type cell_type, int degree,
                const std::vector<std::size_t>& value_shape,
                const xt::xtensor<double, 2>& coeffs,
                const std::vector<std::vector<int>>& entity_dofs,
                const xt::xtensor<double, 3>& base_transformations,
                const xt::xtensor<double, 2>& points,
                const xt::xtensor<double, 2>& M = {},
                mapping::type map_type = mapping::type::identity);

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

  /// @todo Fix unclear description of the returned data layout and
  /// consider using rank 4 tensor [deriv][point][num_res][value_shape].
  /// It is presently inconsistent with other data structures,
  ///
  /// Compute basis values and derivatives at set of points.
  ///
  /// @param[in] nd The order of derivatives, up to and including, to
  /// compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions.
  /// The shape of x is (number of points, geometric dimension).
  /// @return The basis functions (and derivatives). The first entry in
  /// the list is the basis function. Higher derivatives are stored in
  /// triangular (2D) or tetrahedral (3D) ordering, i.e. for the (x,y)
  /// derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2),
  /// (3,0)... The function basix::idx can be used to find the
  /// appropriate derivative. If a vector result is expected, it will be
  /// stacked with all x values, followed by all y-values (and then z,
  /// if any), likewise tensor-valued results will be stacked in index
  /// order.
  xt::xtensor<double, 3> tabulate_new(int nd,
                                      const xt::xarray<double>& x) const;

  /// Direct to memory block tabulation
  /// @param nd Number of derivatives
  /// @param x Points
  /// @param basis_data Memory location to fill
  void tabulate(int nd, const xt::xarray<double>& x, double* basis_data) const;

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
  mapping::type mapping_type() const;

  /// Map function values from the reference to a physical cell
  /// @param U The function values on the reference
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @return The function values on the cell
  xt::xtensor<double, 3>
  map_push_forward(const xt::xtensor<double, 3>& U,
                   const xt::xtensor<double, 3>& J,
                   const tcb::span<const double>& detJ,
                   const xt::xtensor<double, 3>& K) const;

  /// Direct to memory push forward
  /// @param U The function values on the reference
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @param u Memory location to fill
  template <typename T, typename E>
  void map_push_forward_m(const xt::xtensor<T, 3>& U,
                          const xt::xtensor<double, 3>& J,
                          const tcb::span<const double>& detJ,
                          const xt::xtensor<double, 3>& K, E&& u) const;

  /// Map function values from a physical cell to the reference
  /// @param u The function values on the cell
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @return The function values on the reference
  xt::xtensor<double, 3> map_pull_back(const xt::xtensor<double, 3>& u,
                                       const xt::xtensor<double, 3>& J,
                                       const tcb::span<const double>& detJ,
                                       const xt::xtensor<double, 3>& K) const;

  /// @todo Weirdly, the u and U
  /// Map function values from a physical cell to the reference
  /// @param u The function values on the cell
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @param U Memory location to fill
  template <typename T, typename E>
  void map_pull_back_m(const xt::xtensor<T, 3>& u,
                       const xt::xtensor<double, 3>& J,
                       const tcb::span<const double>& detJ,
                       const xt::xtensor<double, 3>& K, E&& U) const;

  /// Get the number of dofs on each topological entity: (vertices,
  /// edges, faces, cell) in that order. For example, Lagrange degree 2
  /// on a triangle has vertices: [1, 1, 1], edges: [1, 1, 1], cell: [0]
  /// The sum of the entity dofs must match the total number of dofs
  /// reported by FiniteElement::dim,
  /// @return List of entity dof counts on each dimension
  const std::vector<std::vector<int>>& entity_dofs() const;

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
  const xt::xtensor<double, 3>& base_transformations() const;

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

private:
  static int compute_value_size(mapping::type map_type, int dim);

  // Cell type
  cell::type _cell_type;

  // The name of the finite element family
  element::family _family;

  // Degree
  int _degree;

  // Value shape
  std::vector<int> _value_shape;

  /// The mapping used to map this element from the reference to a cell
  mapping::type _map_type;

  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by @f$\psi_i = \sum_{k} \phi_{k}
  // \alpha^{i}_{k}@f$, then _coeffs(i, j) = @f$\alpha^i_k@f$. i.e.,
  // _coeffs.row(i) are the expansion coefficients for shape function i
  // (@f$\psi_{i}@f$).
  xt::xtensor<double, 2> _coeffs;

  // Number of dofs associated each subentity
  //
  // The dofs of an element are associated with entities of different
  // topological dimension (vertices, edges, faces, cells). The dofs are
  // listed in this order, with vertex dofs first. Each entry is the dof
  // count on the associated entity, as listed by cell::topology.
  std::vector<std::vector<int>> _entity_dofs;

  // Base transformations
  xt::xtensor<double, 3> _base_transformations;

  // Set of points used for point evaluation
  // Experimental - currently used for an implementation of
  // "tabulate_dof_coordinates" Most useful for Lagrange. This may change or go
  // away. For non-Lagrange elements, these points will be used in combination
  // with _interpolation_matrix to perform interpolation
  xt::xtensor<double, 2> _points;

  /// The interpolation weights and points
  xt::xtensor<double, 2> _matM;

  // The mapping that maps values on the reference to values on a physical cell
  std::function<std::vector<double>(const tcb::span<const double>&,
                                    const xt::xtensor<double, 2>&, const double,
                                    const xt::xtensor<double, 2>&)>
      _map_push_forward;
};

/// Create an element by name
FiniteElement create_element(std::string family, std::string cell, int degree);

/// Create an element by name
FiniteElement create_element(element::family family, cell::type cell,
                             int degree);

/// Return the version number of basix across projects
/// @return version string
std::string version();

//-----------------------------------------------------------------------------
template <typename T, typename E>
void FiniteElement::map_push_forward_m(const xt::xtensor<T, 3>& U,
                                       const xt::xtensor<double, 3>& J,
                                       const tcb::span<const double>& detJ,
                                       const xt::xtensor<double, 3>& K,
                                       E&& u) const
{
  // FIXME: Should U.shape(2) be replaced by the physical value size?
  // Can it differ?
  // std::array<std::size_t, 3> s = {U.shape(0), U.shape(1), U.shape(2)};
  // auto _u = xt::adapt(u, s[0] * s[1] * s[2], xt::no_ownership(), s);

  // Loop over each point
  for (std::size_t p = 0; p < U.shape(0); ++p)
  {
    auto J_p = xt::view(J, p, xt::all(), xt::all());
    auto K_p = xt::view(K, p, xt::all(), xt::all());
    if constexpr (std::is_same<T, double>::value)
    {
      // Loop over values at each point
      for (std::size_t i = 0; i < U.shape(1); ++i)
      {
        // Note: we assign here to a xt::xtensor<double, 1> until the
        // maps are updated to accept xtensor objects rather than spans
        // auto U_data = xt::row(U_b, i);
        xt::xtensor<double, 1> U_data = xt::view(U, p, i, xt::all());
        std::vector<double> f = _map_push_forward(U_data, J_p, detJ[p], K_p);
        for (std::size_t j = 0; j < f.size(); ++j)
          u(p, i, j) = f[j];
      }
    }
    else
    {
      // Note: This is a bit crazy; the maps should handle
      // real/complex/float and whatever else
      throw std::runtime_error("Complex not suppoted.");

      // See https://github.com/xtensor-stack/xtensor/issues/1701
      // for (std::size_t i = 0; i < U_b.shape(0); ++i)
      // {
      //   xt::xtensor<double, 1> U_data_r = xt::real(xt::row(U_b, i));
      //   xt::xtensor<double, 1> U_data_c = xt::imag(xt::row(U_b, i));
      //   std::vector<double> ur = _map_push_forward(U_data_r, J_p, detJ[p],
      //   K_p); std::vector<double> uc = _map_push_forward(U_data_c, J_p,
      //   detJ[p], K_p); for (std::size_t j = 0; j < ur.size(); ++j)
      //     u_b(i, j) = std::complex(ur[j], uc[j]);
      // }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T, typename E>
void FiniteElement::map_pull_back_m(const xt::xtensor<T, 3>& u,
                                    const xt::xtensor<double, 3>& J,
                                    const tcb::span<const double>& detJ,
                                    const xt::xtensor<double, 3>& K,
                                    E&& U) const
{
  // Loop over each point
  for (std::size_t p = 0; p < u.shape(0); ++p)
  {
    auto J_p = xt::view(J, p, xt::all(), xt::all());
    auto K_p = xt::view(K, p, xt::all(), xt::all());
    if constexpr (std::is_same<T, double>::value)
    {
      // Loop over each item at point to be transformed
      for (std::size_t i = 0; i < u.shape(1); ++i)
      {
        // Note: we assign here to a xt::xtensor<double, 1> until the
        // maps are updated to accept xtensor objects rather than spans
        // auto U_data = xt::row(U_b, i);

        // Map data
        xt::xtensor<double, 1> u_data = xt::view(u, p, i, xt::all());
        std::vector<double> f
            = _map_push_forward(u_data, K_p, 1.0 / detJ[p], J_p);
        for (std::size_t j = 0; j < f.size(); ++j)
          U(p, i, j) = f[j];
      }
    }
    else
    {
      // Note: This is a bit crazy; the maps should handle
      // real/complex/float and whatever else
      throw std::runtime_error("Complex not suppoted.");

      // for (int i = 0; i < nresults; ++i)
      // {
      //   Eigen::ArrayXd tmp_r = physical_data.row(pt * nresults + i).real();
      //   Eigen::ArrayXd tmp_c = physical_data.row(pt * nresults + i).imag();
      //   std::vector<double> Ur
      //       = _map_push_forward(tmp_r, current_K, 1 / detJ[pt], current_J);
      //   std::vector<double> Uc
      //       = _map_push_forward(tmp_c, current_K, 1 / detJ[pt], current_J);
      //   for (std::size_t j = 0; j < Ur.size(); ++j)
      //     reference_array(pt * nresults + i, j) = std::complex(Ur[j], Uc[j]);
      // }
    }
  }
}
//-----------------------------------------------------------------------------

} // namespace basix
