
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

// Implementation of a polynomial of N variables as a set of coefficients
// The total number of coefficients determines the order of the polynomial.
// e.g. in 1D, there are n+1 coefficients for an order n polynomial,
// e.g. in 2D there are 6 coefficients for order 2: 1, x, y, x^2, xy, y^2.
class Polynomial
{
public:
  // Default constructor
  Polynomial() : dim(-1), order(-1) {}

  // Static method instantiating an order zero polynomial with value 1.0
  static Polynomial one(int N)
  {
    Polynomial p;
    p.dim = N;
    p.order = 0;
    p.coeffs.resize(1);
    p.coeffs[0] = 1.0;
    return p;
  }

  // Static method instantiating an order one polynomial with value x
  static Polynomial x(int N)
  {
    Polynomial p;
    p.dim = N;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[1] = 1.0;
    return p;
  }

  // Static method instantiating an order one polynomial with value y
  static Polynomial y(int N)
  {
    assert(N > 1);
    Polynomial p;
    p.dim = N;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[2] = 1.0;
    return p;
  }

  // Static method instantiating an order one polynomial with value z
  static Polynomial z(int N)
  {
    assert(N == 3);
    Polynomial p;
    p.dim = N;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[3] = 1.0;
    return p;
  }

  // Add two polynomials
  const Polynomial operator+(const Polynomial& other) const;

  // Subtract two polynomials
  const Polynomial operator-(const Polynomial& other) const;

  // Multiply two polynomials
  const Polynomial operator*(const Polynomial& other) const;

  // Multiply by a scalar
  const Polynomial operator*(const double& scale) const;

  // Multiply by a scalar
  Polynomial& operator*=(const double& scale);

  // Compute polynomial value at points (tabulate)
  Eigen::ArrayXd
  tabulate(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& points) const;

  // Differentiate with respect to x, y or z, returning a polynomial of lower
  // order.
  // @param axis (x=0, y=1, z=2)
  const Polynomial diff(int axis) const;

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
