// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include <Eigen/Dense>
#include <array>
#include <vector>

#pragma once

namespace libtab
{

class FiniteElement
{
  /// Finite Element
  /// The basis is stored as a set of coefficients, which are applied to the
  /// underlying expansion set for that cell type, when tabulating.

public:
  /// A finite element
  FiniteElement(
      cell::Type cell_type, int degree, int value_size,
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          coeffs)
      : _cell_type(cell_type), _degree(degree), _value_size(value_size),
        _coeffs(coeffs), _entity_dofs({0})
  {
  }

  /// Destructor
  ~FiniteElement() = default;

  /// Compute basis values and derivatives at set of points.
  ///
  /// @param[in] nd The order of derivatives, up to and including,
  /// to compute. Use 0 for the basis functions only.
  /// @param[in] x The points at which to compute the basis functions.
  /// The shape of x is (number of points, geometric dimension).
  /// @return The basis functions (and derivatives). The first index is
  /// the basis function. Higher derivatives are stored in triangular (2D)
  /// or tetrahedral (3D) ordering, i.e. for the (x,y) derivatives in
  /// 2D: (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0)... If a vector
  /// result is expected, it will be stacked with all x values, followed
  /// by all y-values (and then z, if any).
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  tabulate(int nd, const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& x) const;

  /// Get the element cell type
  /// @return The cell type
  cell::Type cell_type() const { return _cell_type; }

  /// Get the element polynomial degree
  /// @return Polynomial degree
  int degree() const { return _degree; }

  /// Get the element value size
  /// @return Value size
  int value_size() const { return _value_size; }

  /// Get the number of degrees of freedom
  /// @return Number of degrees of freedom
  int ndofs() const { return _coeffs.rows(); }

  /// Get the dofs -> topological dimension mapping
  /// @return List
  std::array<int, 4> entity_dofs() const { return _entity_dofs; }

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
  /// This function takes the matrices B (span_coeffs) and D (dualmat) as
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
  /// (dualmat) given by applying these to p_0 to p_2 is:
  ///  @f[ \mbox{dualmat} = \begin{bmatrix}
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
  /// of the normal components along each edge. The matrix D (dualmat) given
  /// by applying these to @f$p_0@f$ to @f$p_5@f$ is:
  ///   dualmat = @f[ \begin{bmatrix}
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
  /// @param[in] span_coeffs The matrix B containing the expansion
  /// coefficients defining a polynomial basis spanning the polynomial space
  /// for this element.
  /// @param[in] dualmat The matrix D of values obtained by applying each
  /// functional in the dual set to each expansion polynomial
  /// @param[in] condition_check If set, checks the condition of the matrix
  /// B.D^T and throws an error if it is ill-conditioned.
  /// @return The matrix C of expansion coefficients that define the basis
  /// functions of the finite element space.
  static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_expansion_coefficents(
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& span_coeffs,
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& dualmat,
      bool condition_check = false);

private:
  // Cell type
  cell::Type _cell_type;

  // Degree
  int _degree;

  // Value size
  int _value_size;

  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by @f$\psi_i = \sum_{k} \phi_{k} \alpha^{i}_{k}@f$,
  // then _coeffs(i, j) = @f$\alpha^i_k@f$. i.e., _coeffs.row(i) are the
  // expansion coefficients for shape function i (@f$\psi_{i}@f$).
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _coeffs;

  // Number of dofs in entities
  // The dofs of an element are associated with entities of different
  // topological dimension (vertices, edges, faces, cells). The dofs are listed
  // in this order, with vertex dofs first. This array represents the number of
  // dofs on each entity. e.g. for Lagrange of order 2 on a triangle it is [1,
  // 1, 0, 0], since each vertex has one dofs, each edge has 1 dof. For faces
  // and cells, rather than the number of dofs, the edge size is given, e.g. for
  // a triangular face with 6 dofs, the edge size is 3. For a hexahedral cell
  // with 8 internal dofs, the value would be 2.
  std::array<int, 4> _entity_dofs;
};
} // namespace libtab
