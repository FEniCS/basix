// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial.h"

//-----------------------------------------------------------------------------
const Polynomial Polynomial::operator+(const Polynomial& other) const
{
  assert(this->dim == other.dim);
  Polynomial result = *this;
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
Polynomial& Polynomial::operator+=(const Polynomial& other)
{
  assert(this->dim == other.dim);
  int n = other.coeffs.size();
  int m = this->coeffs.size();
  if (n > m)
  {
    this->coeffs.conservativeResize(n);
    this->coeffs.tail(n - m).setZero();
    this->order = other.order;
  }
  this->coeffs.head(n) += other.coeffs;
  return *this;
}
//-----------------------------------------------------------------------------
const Polynomial Polynomial::operator-(const Polynomial& other) const
{
  assert(this->dim == other.dim);
  Polynomial result = *this;
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
const Polynomial Polynomial::operator*(const double& scale) const
{
  Polynomial result = *this;
  result.coeffs *= scale;
  return result;
};

//-----------------------------------------------------------------------------
Polynomial& Polynomial::operator*=(const double& scale)
{
  this->coeffs *= scale;
  return *this;
}
//-----------------------------------------------------------------------------
const Polynomial Polynomial::operator*(const Polynomial& other) const
{
  assert(this->dim == other.dim);
  Polynomial result;
  result.dim = this->dim;
  int n0 = this->order;
  int n1 = other.order;
  int n = n0 + n1;
  result.order = n;

  // Compute size of product coeff vector
  int m = 1;
  for (int i = 0; i < this->dim; ++i)
  {
    m *= (n + i + 1);
    m /= (i + 1);
  }
  result.coeffs.resize(m);
  result.coeffs.setZero();

  // Index vectors to march through polynomial in correct order
  // and a function to update them
  std::vector<int> c0(this->dim, 0);
  std::vector<int> c1(this->dim, 0);
  std::function<void(std::vector<int>&, int)> _update
      = [&_update](std::vector<int>& c, int s) {
          if (s > 1 and c[s - 1] == c[s - 2])
          {
            c[s - 1] = 0;
            _update(c, s - 1);
          }
          else
            ++c[s - 1];
        };

  // Computation of linear index from index vector, needed to
  // locate correct entry of the sum of two index vectors
  auto _idx = [](std::vector<int>& c) {
    int s = 0;
    int n = c.size() - 1;
    for (int i = 0; i < n + 1; ++i)
    {
      int p = c[n - i];
      int r = 1;
      for (int j = 0; j < i + 1; ++j)
      {
        r *= (p + j);
        r /= (1 + j);
      }
      s += r;
    }
    return s;
  };

  // Iterate through both sets of indices, summing coeffs
  // into the correct place in result.coeffs
  std::vector<int> csum(this->dim, 0);
  for (int i = 0; i < this->coeffs.size(); ++i)
  {
    std::fill(c1.begin(), c1.end(), 0);
    for (int j = 0; j < other.coeffs.size(); ++j)
    {
      for (int k = 0; k < this->dim; ++k)
        csum[k] = c0[k] + c1[k];
      result.coeffs[_idx(csum)] += this->coeffs[i] * other.coeffs[j];
      _update(c1, c1.size());
    }
    _update(c0, c0.size());
  }

  return result;
};
//-----------------------------------------------------------------------------
Eigen::ArrayXd
Polynomial::tabulate(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& points) const
{
  assert((int)points.cols() == dim);
  Eigen::ArrayXd v(points.rows());
  v.setZero();
  const int m = this->order;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p(
      points.rows(), points.cols());
  p.fill(1.0);

  for (int k = 0; k < m + 1; ++k)
  {
    if (dim > 1)
    {
      p.col(1).fill(1.0);
      for (int l = 0; l < m + 1 - k; ++l)
      {
        if (dim == 3)
        {
          p.col(2).fill(1.0);
          for (int q = 0; q < m + 1 - k - l; ++q)
          {
            v += p.col(0) * p.col(1) * p.col(2) * this->coeffs[idx(k, l, q)];
            p.col(2) *= points.col(2);
          }
        }
        else if (dim == 2)
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
double Polynomial::tabulate(double x) const
{
  if (this->dim != 1)
    throw std::runtime_error("Cannot tabulate vector with scalar input");
  const int m = this->order;
  double v = 0.0;
  double p = 1.0;
  for (int k = 0; k < m + 1; ++k)
  {
    v += this->coeffs[k] * p;
    p *= x;
  }
  return v;
}
//-----------------------------------------------------------------------------
const Polynomial Polynomial::diff(const std::vector<int>& d) const
{
  assert((int)d.size() == this->dim);
  Polynomial result = *this;

  if (dim == 0)
  {
    assert(result.coeffs.size() == 1);
    assert(result.order == 0);
    result.coeffs[0] = 0.0;
  }
  else if (dim == 1)
  {
    for (int i = 0; i < d[0]; ++i)
    {
      if (result.order == 0)
        result.coeffs[0] = 0.0;
      else
      {
        for (int k = 0; k < result.order; ++k)
          result.coeffs[k] = (k + 1) * result.coeffs[k + 1];
        --result.order;
      }
    }
    result.coeffs.conservativeResize(result.order + 1);
  }
  else if (dim == 2)
  {
    for (int i = 0; i < d[0]; ++i)
    {
      if (result.order == 0)
        result.coeffs[0] = 0.0;
      else
      {
        for (int k = 0; k < result.order; ++k)
          for (int l = 0; l < result.order - k; ++l)
            result.coeffs[idx(k, l)] = (k + 1) * result.coeffs[idx(k + 1, l)];
        --result.order;
      }
    }
    for (int i = 0; i < d[1]; ++i)
    {
      if (result.order == 0)
        result.coeffs[0] = 0.0;
      else
      {
        for (int k = 0; k < result.order; ++k)
          for (int l = 0; l < result.order - k; ++l)
            result.coeffs[idx(k, l)] = (l + 1) * result.coeffs[idx(k, l + 1)];
        --result.order;
      }
    }
    result.coeffs.conservativeResize((result.order + 1) * (result.order + 2)
                                     / 2);
  }
  else if (dim == 3)
  {
    for (int i = 0; i < d[0]; ++i)
    {
      if (result.order == 0)
        result.coeffs[0] = 0.0;
      else
      {
        for (int k = 0; k < result.order; ++k)
          for (int l = 0; l < result.order - k; ++l)
            for (int q = 0; q < result.order - k - l; ++q)
              result.coeffs[idx(k, l, q)]
                  = (k + 1) * result.coeffs[idx(k + 1, l, q)];
        --result.order;
      }
    }
    for (int i = 0; i < d[1]; ++i)
    {
      if (result.order == 0)
        result.coeffs[0] = 0.0;
      else
      {
        for (int k = 0; k < result.order; ++k)
          for (int l = 0; l < result.order - k; ++l)
            for (int q = 0; q < result.order - k - l; ++q)
              result.coeffs[idx(k, l, q)]
                  = (l + 1) * result.coeffs[idx(k, l + 1, q)];
        --result.order;
      }
    }
    for (int i = 0; i < d[2]; ++i)
    {
      if (result.order == 0)
        result.coeffs[0] = 0.0;
      else
      {
        for (int k = 0; k < result.order; ++k)
          for (int l = 0; l < result.order - k; ++l)
            for (int q = 0; q < result.order - k - l; ++q)
              result.coeffs[idx(k, l, q)]
                  = (q + 1) * result.coeffs[idx(k, l, q + 1)];
        --result.order;
      }
    }
    result.coeffs.conservativeResize((result.order + 1) * (result.order + 2)
                                     * (result.order + 3) / 6);
  }
  else
    throw std::runtime_error("Invalid dimension");

  return result;
}
