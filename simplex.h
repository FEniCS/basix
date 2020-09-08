
#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class ReferenceSimplex
{
 public:
  ReferenceSimplex(int dim);

  // Dimension
  int dim() const
  {
    return _dim;
  }

  // Return the vertex points of the reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
    reference_geometry() const
  {
    return _ref_geom;
  }

  // Create a lattice of points on the simplex
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    lattice(int n) const;

  // Orthonormal polynomial basis on simplex
  std::vector<Polynomial>
    compute_polynomial_set(int n) const;

 private:
  int _dim;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _ref_geom;
};
