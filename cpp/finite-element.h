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
      std::array<int, 4> entity_dofs)
      : _cell_type(cell_type), _degree(degree), _value_size(value_size),
        _coeffs(coeffs), _entity_dofs({entity_dofs})
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
};
} // namespace libtab
