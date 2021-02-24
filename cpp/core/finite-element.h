// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

// FIXME: just include everything for now
// Need to define public API

#pragma once

#include "cell.h"
#include "element-families.h"
#include "mappings.h"
#include <Eigen/Core>
#include <string>
#include <vector>

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
/// @param[in] cell_type The cells shape.
/// @param[in] span_coeffs The matrix B containing the expansion
/// coefficients defining a polynomial basis spanning the polynomial
/// space for this element.
/// @param[in] interpolation_matrix The interpolation matrix
/// @param[in] interpolation_points The interpolation points
/// @param[in] order The degree of the polynomial set
/// @param[in] condition_check If set, checks the condition of the
/// matrix B.D^T and throws an error if it is ill-conditioned.
/// @return The matrix C of expansion coefficients that define the basis
/// functions of the finite element space.
Eigen::MatrixXd
compute_expansion_coefficients(cell::type cell_type,
                               const Eigen::MatrixXd& span_coeffs,
                               const Eigen::MatrixXd& interpolation_matrix,
                               const Eigen::ArrayXXd& interpolation_points,
                               const int order, bool condition_check = false);

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
std::pair<Eigen::ArrayXXd, Eigen::MatrixXd> combine_interpolation_data(
    const Eigen::ArrayXXd& points_1d, const Eigen::ArrayXXd& points_2d,
    const Eigen::ArrayXXd& points_3d, const Eigen::MatrixXd& matrix_1d,
    const Eigen::MatrixXd& matrix_2d, const Eigen::MatrixXd& matrix_3d,
    const int tdim, const int value_size);

/// Finite Element
/// The basis is stored as a set of coefficients, which are applied to the
/// underlying expansion set for that cell type, when tabulating.
class FiniteElement
{

public:
  /// A finite element
  FiniteElement(element::family family, cell::type cell_type, int degree,
                const std::vector<int>& value_shape,
                const Eigen::ArrayXXd& coeffs,
                const std::vector<std::vector<int>>& entity_dofs,
                const std::vector<Eigen::MatrixXd>& base_permutations,
                const Eigen::ArrayXXd& points,
                const Eigen::MatrixXd interpolation_matrix = {},
                mapping::type mapping_type = mapping::type::identity);

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
  /// @return The basis functions (and derivatives). The first entry in
  /// the list is the basis function. Higher derivatives are stored in
  /// triangular (2D) or tetrahedral (3D) ordering, i.e. for the (x,y)
  /// derivatives in 2D: (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0)...
  /// The function basix::idx can be used to find the appropriate
  /// derivative. If a vector result is expected, it will be stacked
  /// with all x values, followed by all y-values (and then z, if any),
  /// likewise tensor-valued results will be stacked in index order.
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const;

  /// Direct to memory block tabulation
  /// @param nd Number of derivatives
  /// @param x Points
  /// @param basis_data Memory location to fill
  void tabulate_to_memory(int nd, const Eigen::ArrayXXd& x,
                          double* basis_data) const;

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
  const mapping::type mapping_type() const;

  /// Map function values from the reference to a physical cell
  /// @param reference_data The function values on the reference
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @return The function values on the cell
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  map_push_forward(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& reference_data,
                   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& J,
                   const Eigen::ArrayXd& detJ,
                   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& K) const;

  /// Direct to memory push forward
  /// @param reference_data The function values on the reference
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @param physical_data Memory location to fill
  template <typename T>
  void
  map_push_forward_m(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& reference_data,
                     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& J,
                     const Eigen::ArrayXd& detJ,
                     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& K,
                     T* physical_data) const;

  /// Map function values from a physical cell to the reference
  /// @param physical_data The function values on the cell
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @return The function values on the reference
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  map_pull_back(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& physical_data,
                const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& J,
                const Eigen::ArrayXd& detJ,
                const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& K) const;

  /// Map function values from a physical cell to the reference
  /// @param physical_data The function values on the cell
  /// @param J The Jacobian of the mapping
  /// @param detJ The determinant of the Jacobian of the mapping
  /// @param K The inverse of the Jacobian of the mapping
  /// @param reference_data Memory location to fill
  template <typename T>
  void map_pull_back_m(
      const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& physical_data,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& J,
      const Eigen::ArrayXd& detJ,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& K,
      T* reference_data) const;

  /// Get the number of dofs on each topological entity: (vertices,
  /// edges, faces, cell) in that order. For example, Lagrange degree 2
  /// on a triangle has vertices: [1, 1, 1], edges: [1, 1, 1], cell: [0]
  /// The sum of the entity dofs must match the total number of dofs
  /// reported by FiniteElement::dim,
  /// @return List of entity dof counts on each dimension
  const std::vector<std::vector<int>>& entity_dofs() const;

  /// Get the base permutations
  /// The base permutations represent the effect of rotating or reflecting
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
  /// For this element, the base permutations are:
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
  /// For this element, the base permutation matrices are:
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
  /// For these DOFs, the subblocks of the base permutation matrices are:
  /// ~~~~~~~~~~~~~~~~
  ///   rotation: [[-1, 1],
  ///              [ 1, 0]]
  ///   reflection: [[0, 1],
  ///                [1, 0]]
  /// ~~~~~~~~~~~~~~~~
  std::vector<Eigen::MatrixXd> base_permutations() const;

  /// Return a set of interpolation points
  const Eigen::ArrayXXd& points() const;

  /// Return the number of interpolation points
  int num_points() const;

  /// Return a matrix of weights interpolation
  /// To interpolate a function in this finite element, the functions
  /// should be evaluated at each point given by
  /// FiniteElement::points(). These function values should then be
  /// multiplied by the weight matrix to give the coefficients of the
  /// interpolated function.
  const Eigen::MatrixXd& interpolation_matrix() const;

private:
  static int compute_value_size(mapping::type mapping_type, int dim);

  // Cell type
  cell::type _cell_type;

  // The name of the finite element family
  element::family _family;

  // Degree
  int _degree;

  // Value shape
  std::vector<int> _value_shape;

  /// The mapping used to map this element from the reference to a cell
  mapping::type _mapping_type;

  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by @f$\psi_i = \sum_{k} \phi_{k}
  // \alpha^{i}_{k}@f$, then _coeffs(i, j) = @f$\alpha^i_k@f$. i.e.,
  // _coeffs.row(i) are the expansion coefficients for shape function i
  // (@f$\psi_{i}@f$).
  Eigen::MatrixXd _coeffs;

  // Number of dofs associated each subentity
  //
  // The dofs of an element are associated with entities of different
  // topological dimension (vertices, edges, faces, cells). The dofs are
  // listed in this order, with vertex dofs first. Each entry is the dof
  // count on the associated entity, as listed by cell::topology.
  std::vector<std::vector<int>> _entity_dofs;

  // Base permutations
  std::vector<Eigen::MatrixXd> _base_permutations;

  // Set of points used for point evaluation
  // Experimental - currently used for an implementation of
  // "tabulate_dof_coordinates" Most useful for Lagrange. This may change or go
  // away. For non-Lagrange elements, these points will be used in combination
  // with _interpolation_matrix to perform interpolation
  Eigen::ArrayXXd _points;

  /// The interpolation weights and points
  Eigen::MatrixXd _interpolation_matrix;

  // The mapping that maps values on the reference to values on a
  // physical cell
  std::function<Eigen::ArrayXd(const Eigen::ArrayXd&, const Eigen::MatrixXd&,
                               const double, const Eigen::MatrixXd&)>
      _map_push_forward;
};

template <typename T>
void FiniteElement::map_push_forward_m(
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        reference_data,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        J,
    const Eigen::ArrayXd& detJ,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        K,
    T* physical_data) const
{
  const int reference_dim = cell::topological_dimension(_cell_type);
  const int physical_dim = J.cols() / reference_dim;
  const int physical_value_size
      = compute_value_size(_mapping_type, physical_dim);
  const int reference_value_size = value_size();
  const int nresults = reference_data.cols() / reference_value_size;
  const int npoints = reference_data.rows();

  for (int pt = 0; pt < npoints; ++pt)
  {
    Eigen::Map<
        const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        reference_block(reference_data.row(pt).data(), reference_value_size,
                        nresults);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        physical_block(physical_data + pt * physical_value_size * nresults,
                       physical_value_size, nresults);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_J(J.row(pt).data(), physical_dim, reference_dim);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_K(K.row(pt).data(), reference_dim, physical_dim);
    for (int i = 0; i < reference_block.cols(); ++i)
    {
      if constexpr (std::is_same<T, double>::value)
      {
        for (int i = 0; i < reference_block.cols(); ++i)
          physical_block.col(i) = _map_push_forward(
              reference_block.col(i), current_J, detJ[pt], current_K);
      }
      else
      {
        physical_block.col(i).real() = _map_push_forward(
            reference_block.col(i).real(), current_J, detJ[pt], current_K);
        physical_block.col(i).imag() = _map_push_forward(
            reference_block.col(i).imag(), current_J, detJ[pt], current_K);
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void FiniteElement::map_pull_back_m(
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& physical_data,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        J,
    const Eigen::ArrayXd& detJ,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        K,
    T* reference_data) const
{
  const int reference_dim = cell::topological_dimension(_cell_type);
  const int physical_dim = J.cols() / reference_dim;
  const int physical_value_size
      = compute_value_size(_mapping_type, physical_dim);
  const int reference_value_size = value_size();
  const int nresults = physical_data.cols() / physical_value_size;
  const int npoints = physical_data.rows();

  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      reference_array(reference_data, nresults * npoints, reference_value_size);

  for (int pt = 0; pt < npoints; ++pt)
  {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_J(J.row(pt).data(), physical_dim, reference_dim);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        current_K(K.row(pt).data(), reference_dim, physical_dim);
    for (int i = 0; i < nresults; ++i)
    {
      if constexpr (std::is_same<T, double>::value)
      {
        reference_array.row(pt * nresults + i)
            = _map_push_forward(physical_data.row(pt * nresults + i), current_K,
                                1 / detJ[pt], current_J);
      }
      else
      {
        reference_array.row(pt * nresults + i).real()
            = _map_push_forward(physical_data.row(pt * nresults + i).real(),
                                current_K, 1 / detJ[pt], current_J);
        reference_array.row(pt * nresults + i).imag()
            = _map_push_forward(physical_data.row(pt * nresults + i).imag(),
                                current_K, 1 / detJ[pt], current_J);
      }
    }
  }
}
//-----------------------------------------------------------------------------

/// Create an element by name
FiniteElement create_element(std::string family, std::string cell, int degree);

/// Create an element by name
FiniteElement create_element(element::family family, cell::type cell,
                             int degree);

/// Return the version number of basix across projects
/// @return version string
std::string version();

} // namespace basix
