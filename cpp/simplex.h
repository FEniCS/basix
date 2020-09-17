// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class ReferenceSimplex
{
public:
  /// Return the vertex points of the reference element
  /// @param dim
  /// @return Set of vertices
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_simplex(int dim);

  /// Create a lattice of points on the simplex defined by the given vertices
  /// optionally including the exterior points
  /// @param vertices Set ov simplex vertices
  /// @param n number of points in each direction
  /// @param exterior If set, includes outer boundaries
  /// @return Set of points
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_lattice(
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& vertices,
      int n, bool exterior);

  /// Sub-entity of a simplex, given by topological dimension and index
  /// @param simplex Imput set of vertices
  /// @param dim Dimension of sub-entity
  /// @param index Local index of sub-entity
  /// @return Set of points
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  sub(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& simplex,
      int dim, int index);

};
