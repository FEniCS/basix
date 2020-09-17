// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#pragma once

// Compute indexing in a 2D triangular array compressed into a 1D array
static inline int idx(int p, int q) { return (p + q + 1) * (p + q) / 2 + q; }

// Compute indexing in a 3D tetrahedral array compressed into a 1D array
static inline int idx(int p, int q, int r)
{
  return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) / 6
         + (q + r) * (q + r + 1) / 2 + r;
};

class Polynomial
{
  /// Implementation of a polynomial of dim=N variables as a set of coefficients
  /// The total number of coefficients determines the order of the polynomial.
  /// e.g. in 1D (dim=1), there are n+1 coefficients for an order n polynomial,
  /// e.g. in 2D (dim=2), there are 6 coefficients for order 2: 1, x, y, x^2,
  /// xy, y^2.

public:
  /// Default constructor
  Polynomial() : dim(0), order(0), coeffs(1) { coeffs[0] = 0.0; }

  // Constructor with values
  Polynomial(const std::vector<double>& values, int dimension)
      : dim(dimension), order(0), coeffs(values.size())
  {
    std::copy(values.data(), values.data() + values.size(), coeffs.data());
    if (dim == 0)
      assert(values.size() == 1);
    else if (dim == 1)
      order = values.size() - 1;
    else if (dim == 2)
    {
      std::size_t m = 1;
      while (m < values.size())
      {
        ++order;
        m = (order + 1) * (order + 2) / 2;
      }
      if (m != values.size())
        throw std::runtime_error("Incorrect number of initialisers");
    }
    else if (dim == 3)
    {
      std::size_t m = 1;
      while (m < values.size())
      {
        ++order;
        m = (order + 1) * (order + 2) * (order + 3) / 6;
      }
      if (m != values.size())
        throw std::runtime_error("Incorrect number of initialisers");
    }
  }

  /// Static method instantiating an order zero polynomial with value 0.0
  static Polynomial zero() { return Polynomial({0.0}, 0); }

  /// Static method instantiating an order zero polynomial with value 1.0
  static Polynomial one() { return Polynomial({1.0}, 0); }

  /// Static method instantiating an order one polynomial with value x
  /// @param N spatial dimension
  static Polynomial x(int N)
  {
    std::vector<double> p(N + 1);
    p[1] = 1.0;
    return Polynomial(p, N);
  }

  /// Static method instantiating an order one polynomial with value y
  /// @param N spatial dimension
  static Polynomial y(int N)
  {
    assert(N > 1);
    std::vector<double> p(N + 1);
    p[2] = 1.0;
    return Polynomial(p, N);
  }

  /// Static method instantiating an order one polynomial with value z
  /// @param N spatial dimension
  static Polynomial z(int N)
  {
    assert(N == 3);
    return Polynomial({0.0, 0.0, 0.0, 1.0}, N);
  }

  /// Add two polynomials
  const Polynomial operator+(const Polynomial& other) const;

  /// Add two polynomials
  Polynomial& operator+=(const Polynomial& other);

  /// Subtract two polynomials
  const Polynomial operator-(const Polynomial& other) const;

  /// Multiply two polynomials
  const Polynomial operator*(const Polynomial& other) const;

  /// Multiply polynomial by a scalar
  const Polynomial operator*(const double& scale) const;

  // Multiply by a scalar
  Polynomial& operator*=(const double& scale);

  /// Compute polynomial value at points (tabulate)
  /// @param points
  /// @return Polynomial values at points
  Eigen::ArrayXd
  tabulate(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& points) const;

  /// Simple evaluation for dim=1 polynomial at a point x
  /// @param x Point
  /// @return Polynomial value at x
  double tabulate(double x) const;

  /// Differentiate with respect to x, y or z, returning a polynomial of lower
  /// order.
  /// @param d level of differentiation on each axis
  const Polynomial diff(const std::vector<int>& d) const;

  /// Polynomial dimension (number of variables)
  int dimension() const { return dim; }

private:
  // Polynomial dimension (1=x, 2=(x,y), 3=(x,y,z)) etc.
  // Dimension over 3 is not fully supported
  int dim;
  // Polynomial order, somewhat redundant, since coeffs size is related
  // e.g. for dim=2, coeffs.size() = (order + 1) * (order + 2)/2
  int order;
  // Coefficients of polynomial in a triangular array, compressed to a linear
  // Order is e.g. 1, x, y, x^2, xy, y^2, x^3, x^2y, y^2x, y^3 etc. for dim=2
  Eigen::Array<double, Eigen::Dynamic, 1> coeffs;
};
