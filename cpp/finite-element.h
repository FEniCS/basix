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
          coeffs,
      Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          base_permutations)
      : _cell_type(cell_type), _degree(degree), _value_size(value_size),
        _coeffs(coeffs), _entity_dofs({0}),
        _base_permutations(base_permutations), _contains_vectors(false)
  {
  }
  FiniteElement(
      cell::Type cell_type, int degree, int value_size,
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          coeffs,
      Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          base_permutations,
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
          direction_correction)
      : _cell_type(cell_type), _degree(degree), _value_size(value_size),
        _coeffs(coeffs), _entity_dofs({0}),
        _base_permutations(base_permutations), _contains_vectors(true),
        _direction_correction(direction_correction)
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
  /// the derivative. Higher derivatives are stored in triangular (2D)
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

  // FIXME: document better and explain mathematically
  // Applies nodal constraints from dualmat to original
  // coeffs on basis, and return new coeffs.
  static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  apply_dualmat_to_basis(
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& coeffs,
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& dualmat);

  /// Get the base permutations
  /// The base permutations represent the effect of rotating or reflecting
  /// a subentity of the cell on the numbering of the DOFs.
  /// This returns a matrix with one row for each subentity permutation in
  /// the following order:
  ///   Reversing edge 0, reversing edge 1, ...
  ///   Rotate face 0, reflect face 0, rotate face 1, reflect face 1, ...
  ///
  /// Example: Order 3 Lagrange on a triangle
  /// ---------------------------------------
  /// This space has 10 dofs arranged like:
  /// 2
  /// |\
  /// 6 4
  /// |  \
  /// 5 9 3
  /// |    \
  /// 0-7-8-1
  /// For this element, the base permutations are:
  ///   [[0, 1, 2, 4, 3, 5, 6, 7, 8, 9],
  ///    [0, 1, 2, 3, 4, 6, 5, 7, 8, 9]
  ///    [0, 1, 2, 3, 4, 5, 6, 8, 7, 9]]
  /// The first row shows the effect of reversing the diagonal edge. The
  /// second row shows the effect of reversing the vertical edge. The third
  /// row shows the effect of reversing the horizontal edge.
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  base_permutations() const
  {
    return _base_permutations;
  };

  /// Does the space contain vectors?
  /// If true, the space contains vector-valued functions, so direction
  /// correction will need to be applied.
  bool contains_vectors() const { return _contains_vectors; };

  /// Direction correction
  /// The direction correction represents the effect of rotating or reflecting
  /// a subentity of the cell on the direction of vector-values DOFs.
  /// This returns a std::vector of matrices. Each matrix represents the effect
  /// of a subentity permutation. Ther are contained in the std::vector in the
  /// following order:
  ///   Reversing edge 0, reversing edge 1, ...
  ///   Rotate face 0, reflect face 0, rotate face 1, reflect face 1, ...
  ///
  /// Example: Order 1 Raviart-Thomas on a triangle
  /// ---------------------------------------------
  /// This space has 3 dofs arranged like:
  ///   |\
  ///   | \
  ///   |  \
  /// <-1   0
  ///   |  / \
  ///   | L ^ \
  ///   |   |  \
  ///    ---2---
  /// These DOFs are integrals of normal components over the edges: DOFs 0 and 2
  /// are oriented inward, DOF 1 is oriented outwards.
  /// For this element, the direction correction matrices are:
  ///   0: [[-1, 0, 0],
  ///       [ 0, 1, 0],
  ///       [ 0, 0, 1]]
  ///   1: [[1,  0, 0],
  ///       [0, -1, 0],
  ///       [0,  0, 1]]
  ///   2: [[1, 0,  0],
  ///       [0, 1,  0],
  ///       [0, 0, -1]]
  /// The first matrix reverses DOF 0 (as this is on the first edge). The second
  /// matrix reverses DOF 1 (as this is on the second edge). The third matrix
  /// reverses DOF 2 (as this is on the third edge).
  ///
  /// Example: DOFs on the face of Order 2 Nedelec first kind on a tetrahedron
  /// ------------------------------------------------------------------------
  /// On a face of this tetrahedron, this space has two face tangent DOFs:
  /// |\        |\
  /// | \       | \
  /// |  \      | ^\
  /// |   \     | | \
  /// | 0->\    | 1  \
  /// |     \   |     \
  ///  ------    ------
  /// For these DOFs, the subblocks of the direction correction matrices are:
  ///   rotation: [[-1, 1],
  ///              [ 1, 0]]
  ///   reflection: [[0, 1],
  ///                [1, 0]]
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  direction_correction() const
  {
    return _direction_correction;
  };

private:
  // Cell type
  cell::Type _cell_type;

  // Degree
  int _degree;

  // Value size
  int _value_size;

  // FIXME: Check shape/layout
  // Shape function coefficient of expansion sets on cell. If shape
  // function is given by \psi_i = \sum_{k} \phi_{k} \alpha^{i}_{k},
  // then _coeffs(i, j) = \alpha^{i}_{k}. I.e., _coeffs.row(i) are the
  // expansion coefficients for shape function i (\psi_{i}).
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

  // Base permutations
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _base_permutations;

  // Does the space contain vectors
  bool _contains_vectors;

  // Direction correction
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      _direction_correction;
};
} // namespace libtab
