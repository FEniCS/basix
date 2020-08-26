
#include <Eigen/Dense>
#include <vector>

#pragma once

static inline std::tuple<double, double, double> jrc(int a, int n)
{
  double an = (a + 2 * n + 1) * (a + 2 * n + 2)
              / static_cast<double>(2 * (n + 1) * (a + n + 1));
  double bn = a * a * (a + 2 * n + 1)
              / static_cast<double>(2 * (n + 1) * (a + n + 1) * (a + 2 * n));
  double cn = n * (a + n) * (a + 2 * n + 2)
              / static_cast<double>((n + 1) * (a + n + 1) * (a + 2 * n));
  return std::tuple<double, double, double>(an, bn, cn);
}

static inline int idx(int p, int q) { return (p + q + 1) * (p + q) / 2 + q; }

static inline int idx(int p, int q, int r)
{
  return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) / 6
         + (q + r) * (q + r + 1) / 2 + r;
};

template <int N>
class PolyN
{
public:
  PolyN()
  {
    if (N < 1 or N > 3)
      throw std::runtime_error("Invalid dimension (must be in range 1-3)");
  }

  static PolyN one()
  {
    PolyN p;
    p.order = 0;
    p.coeffs.resize(1);
    p.coeffs[0] = 1.0;
    return p;
  }
  
  static PolyN x()
  {
    PolyN p;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[1] = 1.0;
    return p;
  }

  static PolyN y()
  {
    assert (N > 1);
    PolyN p;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[2] = 1.0;
    return p;
  }

  static PolyN z()
  {
    assert (N == 3);
    PolyN p;
    p.order = 1;
    p.coeffs.resize(N + 1);
    p.coeffs.setZero();
    p.coeffs[3] = 1.0;
    return p;
  }

  // Add two polynomials
  const PolyN operator+(const PolyN& other) const;

  // Subtract two polynomials
  const PolyN operator-(const PolyN& other) const;

  // Multiply two polynomials
  const PolyN operator*(const PolyN& other) const;

  // Multiply by a scalar
  const PolyN operator*(const double& scale) const;

  // Multiply by a scalar
  PolyN& operator*=(const double& scale);

  // Compute polynomial value at points
  Eigen::ArrayXd
  tabulate(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& points);

  // Differentiate
  const PolyN diff(int axis) const;
  
private:
  int order;
  Eigen::Array<double, Eigen::Dynamic, 1> coeffs;
};

//-----------------------------------------------------------------------------
template <int N>
const PolyN<N> PolyN<N>::operator+(const PolyN<N>& other) const
{
  PolyN<N> result = *this;
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
const PolyN<N> PolyN<N>::operator-(const PolyN<N>& other) const
{
  PolyN<N> result = *this;
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
const PolyN<N> PolyN<N>::operator*(const double& scale) const
{
  PolyN<N> result = *this;
  result.coeffs *= scale;
  return result;
};

//-----------------------------------------------------------------------------
template <int N>
PolyN<N>& PolyN<N>::operator*=(const double& scale)
{
  this->coeffs *= scale;
  return *this;
}

//-----------------------------------------------------------------------------
template <int N>
Eigen::ArrayXd
PolyN<N>::tabulate(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& points)
{
  assert(points.cols() == N);
  Eigen::ArrayXd v(points.rows());
  v.setZero();
  const int m = this->order;

  for (int i = 0; i < points.rows(); ++i)
  {
    double xx = 1.0;
    for (int k = 0; k < m + 1; ++k)
    {
      if (N > 1)
      {
        double yy = 1.0;
        for (int l = 0; l < m + 1 - k; ++l)
        {
          if (N == 3)
          {
            double zz = 1.0;
            for (int q = 0; q < m + 1 - k - l; ++q)
            {
              v[i] += xx * yy * zz * this->coeffs[idx(k, l, q)];
              zz *= points(i, 2);
            }
          }
          else if (N == 2)
            v[i] += xx * yy * this->coeffs[idx(k, l)];
          yy *= points(i, 1);
        }
      }
      else
        v[i] += xx * this->coeffs[k];
      xx *= points(i, 0);
    }
  }
  return v;
}

//-----------------------------------------------------------------------------
template <>
const PolyN<3> PolyN<3>::operator*(const PolyN<3>& other) const
{
  PolyN<3> result;
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
template <>
const PolyN<2> PolyN<2>::operator*(const PolyN<2>& other) const
{
  PolyN<2> result;
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
template <>
const PolyN<1> PolyN<1>::operator*(const PolyN<1>& other) const
{
  PolyN<1> result;
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
template <int N>
const PolyN<N> PolyN<N>::diff(int axis) const {

  assert (axis >= 0);
  PolyN<N> result;
  const int m = this->order;
  result.order = m - 1;

  if (N == 1)
    {
      assert (axis == 0);
      result.coeffs.resize(m);
      for (int k = 0; k < m; ++k)
	result.coeffs[k] = (k + 1) * this->coeffs[k + 1];
    }

  if (N == 2)
    {
      assert (axis < 2);
      result.coeffs.resize(m *(m + 1)/2);
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
      assert (axis < 3);
      result.coeffs.resize(m *(m + 1)*(m + 2)/6);
      if (axis == 0)
	{
	  for (int k = 0; k < m; ++k)
	    for (int l = 0; l < m - k; ++l)
	      for (int q = 0; q < m - k - l; ++q)
		result.coeffs[idx(k, l, q)] = (k + 1) * this->coeffs[idx(k + 1, l, q)];
	}
      else if (axis == 1)
	{
	  for (int k = 0; k < m; ++k)
	    for (int l = 0; l < m - k; ++l)
	      for (int q = 0; q < m - k - l; ++q)
		result.coeffs[idx(k, l, q)] = (l + 1) * this->coeffs[idx(k, l + 1, q)];
	}
      else
	{
	  for (int k = 0; k < m; ++k)
	    for (int l = 0; l < m - k; ++l)
	      for (int q = 0; q < m - k - l; ++q)
		result.coeffs[idx(k, l, q)] = (q + 1) * this->coeffs[idx(k, l, q + 1)];
	}
    }
  
  return result;
}
