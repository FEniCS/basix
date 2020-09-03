
#include <Eigen/Dense>
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
template <int N>
class Polynomial
{
public:
  // Default constructor
  Polynomial()
  {
    if (N < 1 or N > 3)
      throw std::runtime_error("Invalid dimension (must be in range 1-3)");
  }

  // Static method instantiating an order zero polynomial with value 1.0
  static Polynomial one()
  {
    Polynomial p;
    p.order = 0;
    p.coeffs.resize(1);
    p.coeffs[0] = 1.0;
    return p;
  }

  // Static method instantiating an order one polynomial with value x
  static Polynomial x()
  {
    Polynomial p;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[1] = 1.0;
    return p;
  }

  // Static method instantiating an order one polynomial with value y
  static Polynomial y()
  {
    assert(N > 1);
    Polynomial p;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[2] = 1.0;
    return p;
  }

  // Static method instantiating an order one polynomial with value z
  static Polynomial z()
  {
    assert(N == 3);
    Polynomial p;
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
  int order;
  Eigen::Array<double, Eigen::Dynamic, 1> coeffs;
};

//-----------------------------------------------------------------------------
template <int N>
const Polynomial<N> Polynomial<N>::operator+(const Polynomial<N>& other) const
{
  Polynomial<N> result = *this;
  int n = other.coeffs.size();
  int m = result.coeffs.size();
  if (n > m)
  {
    result.coeffs.conservativeResize(n);
    result.coeffs.tail(n - m).setZero();
    result.order = other.order;
  }
  result.coeffs.head(n) += other.coeffs;
  return result;
};

//-----------------------------------------------------------------------------
template <int N>
const Polynomial<N> Polynomial<N>::operator-(const Polynomial<N>& other) const
{
  Polynomial<N> result = *this;
  int n = other.coeffs.size();
  int m = result.coeffs.size();
  if (n > m)
  {
    result.coeffs.conservativeResize(n);
    result.coeffs.tail(n - m).setZero();
    result.order = other.order;
  }
  result.coeffs.head(n) -= other.coeffs;
  return result;
};

//-----------------------------------------------------------------------------
template <int N>
const Polynomial<N> Polynomial<N>::operator*(const double& scale) const
{
  Polynomial<N> result = *this;
  result.coeffs *= scale;
  return result;
};

//-----------------------------------------------------------------------------
template <int N>
Polynomial<N>& Polynomial<N>::operator*=(const double& scale)
{
  this->coeffs *= scale;
  return *this;
}

//-----------------------------------------------------------------------------
template <int N>
Eigen::ArrayXd Polynomial<N>::tabulate(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        points) const
{
  assert(points.cols() == N);
  Eigen::ArrayXd v(points.rows());
  v.setZero();
  const int m = this->order;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
      points.rows(), points.cols());
  p.fill(1.0);

  for (int k = 0; k < m + 1; ++k)
  {
    if (N > 1)
    {
      p.col(1).fill(1.0);
      for (int l = 0; l < m + 1 - k; ++l)
      {
        if (N == 3)
        {
	  p.col(2).fill(1.0);
          for (int q = 0; q < m + 1 - k - l; ++q)
          {
            v += p.col(0) * p.col(1) * p.col(2) * this->coeffs[idx(k, l, q)];
            p.col(2) *= points.col(2);
          }
        }
        else if (N == 2)
          v += p.col(0) * p.col(1) * this->coeffs[idx(k, l)];
        p.col(1) *= points.col(1);
      }
    }
    else
      v += p.col(0) * this->coeffs[k];
    p.col(0) *= points.col(0);
  }

  return v;
}

//-----------------------------------------------------------------------------
// Specialisation for multiplying together 3D polynomials
template <>
const Polynomial<3> Polynomial<3>::operator*(const Polynomial<3>& other) const
{
  Polynomial<3> result;
  int n0 = this->order;
  int n1 = other.order;
  int n = n0 + n1;
  result.order = n;
  int m = (n + 3) * (n + 2) * (n + 1) / 6;
  result.coeffs.resize(m);
  result.coeffs.setZero();

  for (int p0 = 0; p0 < n0 + 1; ++p0)
    for (int q0 = 0; q0 < n0 + 1 - p0; ++q0)
      for (int r0 = 0; r0 < n0 + 1 - p0 - q0; ++r0)
      {
        int i0 = idx(p0, q0, r0);
        for (int p1 = 0; p1 < n1 + 1; ++p1)
          for (int q1 = 0; q1 < n1 + 1 - p1; ++q1)
            for (int r1 = 0; r1 < n1 + 1 - p1 - q1; ++r1)
            {
              int i1 = idx(p1, q1, r1);
              int i2 = idx(p0 + p1, q0 + q1, r0 + r1);
              result.coeffs[i2] += this->coeffs[i0] * other.coeffs[i1];
            }
      }
  return result;
};

//-----------------------------------------------------------------------------
// Specialisation for multiplying together 2D polynomials
template <>
const Polynomial<2> Polynomial<2>::operator*(const Polynomial<2>& other) const
{
  Polynomial<2> result;
  int n0 = this->order;
  int n1 = other.order;
  int n = n0 + n1;
  result.order = n;
  int m = (n + 2) * (n + 1) / 2;
  result.coeffs.resize(m);
  result.coeffs.setZero();

  for (int p0 = 0; p0 < n0 + 1; ++p0)
    for (int q0 = 0; q0 < n0 + 1 - p0; ++q0)
    {
      int i0 = idx(p0, q0);
      for (int p1 = 0; p1 < n1 + 1; ++p1)
        for (int q1 = 0; q1 < n1 + 1 - p1; ++q1)
        {
          int i1 = idx(p1, q1);
          int i2 = idx(p0 + p1, q0 + q1);
          result.coeffs[i2] += this->coeffs[i0] * other.coeffs[i1];
        }
    }
  return result;
};

//-----------------------------------------------------------------------------
// Specialisation for multiplying together 1D polynomials
template <>
const Polynomial<1> Polynomial<1>::operator*(const Polynomial<1>& other) const
{
  Polynomial<1> result;
  int n0 = this->order;
  int n1 = other.order;
  int n = n0 + n1;
  result.order = n;
  int m = (n + 1);
  result.coeffs.resize(m);
  result.coeffs.setZero();

  for (int p0 = 0; p0 < n0 + 1; ++p0)
    for (int p1 = 0; p1 < n1 + 1; ++p1)
      result.coeffs[p0 + p1] += this->coeffs[p0] * other.coeffs[p1];

  return result;
};
//-----------------------------------------------------------------------------
// Differentiation (untested)
template <int N>
const Polynomial<N> Polynomial<N>::diff(int axis) const
{

  assert(axis >= 0);
  Polynomial<N> result;
  const int m = this->order;
  result.order = m - 1;

  if (N == 1)
  {
    assert(axis == 0);
    result.coeffs.resize(m);
    for (int k = 0; k < m; ++k)
      result.coeffs[k] = (k + 1) * this->coeffs[k + 1];
  }

  if (N == 2)
  {
    assert(axis < 2);
    result.coeffs.resize(m * (m + 1) / 2);
    if (axis == 0)
    {
      for (int k = 0; k < m; ++k)
        for (int l = 0; l < m - k; ++l)
          result.coeffs[idx(k, l)] = (k + 1) * this->coeffs[idx(k + 1, l)];
    }
    else
    {
      for (int k = 0; k < m; ++k)
        for (int l = 0; l < m - k; ++l)
          result.coeffs[idx(k, l)] = (l + 1) * this->coeffs[idx(k, l + 1)];
    }
  }

  if (N == 3)
  {
    assert(axis < 3);
    result.coeffs.resize(m * (m + 1) * (m + 2) / 6);
    if (axis == 0)
    {
      for (int k = 0; k < m; ++k)
        for (int l = 0; l < m - k; ++l)
          for (int q = 0; q < m - k - l; ++q)
            result.coeffs[idx(k, l, q)]
                = (k + 1) * this->coeffs[idx(k + 1, l, q)];
    }
    else if (axis == 1)
    {
      for (int k = 0; k < m; ++k)
        for (int l = 0; l < m - k; ++l)
          for (int q = 0; q < m - k - l; ++q)
            result.coeffs[idx(k, l, q)]
                = (l + 1) * this->coeffs[idx(k, l + 1, q)];
    }
    else
    {
      for (int k = 0; k < m; ++k)
        for (int l = 0; l < m - k; ++l)
          for (int q = 0; q < m - k - l; ++q)
            result.coeffs[idx(k, l, q)]
                = (q + 1) * this->coeffs[idx(k, l, q + 1)];
    }
  }

  return result;
}
