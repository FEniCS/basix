
#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class ReferenceSimplex
{
 public:
  ReferenceSimplex(int dim);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    reference_geometry()
  {
    return _ref_geom;
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    lattice(int n);

  // Orthonormal polynomial basis on simplex
  std::vector<Polynomial>
    compute_polynomial_set(int n);

 private:
  int _dim;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _ref_geom;
};
