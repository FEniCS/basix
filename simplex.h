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
  // Return the vertex points of the reference element
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_simplex(int dim);

  // Create a lattice of points on the simplex defined by the given vertices
  // optionally including the exterior points
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  create_lattice(
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& vertices,
      int n, bool exterior);

  // Sub-entity of a simplex, given by topological dimension and index
  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  sub(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& simplex,
      int dim, int index);

  // Orthonormal polynomial basis on reference simplex of dimension dim
  static std::vector<Polynomial> compute_polynomial_set(int dim, int n);
};
