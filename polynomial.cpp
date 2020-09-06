
#include "polynomial.h"

//-----------------------------------------------------------------------------
const Polynomial Polynomial::operator+(const Polynomial& other) const
{
  assert (this->dim == other.dim);
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
const Polynomial Polynomial::operator-(const Polynomial& other) const
{
  assert (this->dim == other.dim);
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
Eigen::ArrayXd Polynomial::tabulate(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        points) const
{
  assert(points.cols() == dim);
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

// Differentiation (untested)
const Polynomial Polynomial::diff(int axis) const
{
  assert(axis >= 0);
  Polynomial result;
  result.dim = this->dim;
  const int m = this->order;
  result.order = m - 1;

  if (dim == 1)
  {
    assert(axis == 0);
    result.coeffs.resize(m);
    for (int k = 0; k < m; ++k)
      result.coeffs[k] = (k + 1) * this->coeffs[k + 1];
  }

  if (dim == 2)
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

  if (dim == 3)
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

const Polynomial Polynomial::operator*(const Polynomial& other) const
{
  Polynomial result;
  result.dim = this->dim;
  int n0 = this->order;
  int n1 = other.order;
  int n = n0 + n1;
  result.order = n;
  int m = 1;
  int m0 = 1;
  int m1 = 1;
  for (int i = 0; i < dim; ++i)
  {
    m0 *= (n0 + i + 1);
    m1 *= (n1 + i + 1);
    m *= (n + i + 1);
    m0 /= (i + 1);
    m1 /= (i + 1);
    m /= (i + 1);
  }
  std::cout << "m = " << m0 << "," << m1 << "," << m << "\n";
  result.coeffs.resize(m);
  result.coeffs.setZero();

  std::vector<int> c0(dim, 0);
  std::vector<int> c1(dim, 0);
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

  std::vector<int> csum(dim, 0);
  for (int i = 0; i < m0; ++i)
  {
    std::fill(c1.begin(), c1.end(), 0);
    for (int j = 0; j < m1; ++j)
    {
      for (int k = 0; k < dim; ++k)
        csum[k] = c0[k] + c1[k];
      result.coeffs[_idx(csum)] += this->coeffs[i] * other.coeffs[j];
      std::cout << _idx(csum) << "\n";
      _update(c1, c1.size());
    }
    _update(c0, c0.size());
  }

  return result;
  
  // for (int p0 = 0; p0 < n0 + 1; ++p0)
  //   for (int q0 = 0; q0 < n0 + 1 - p0; ++q0)
  //     for (int r0 = 0; r0 < n0 + 1 - p0 - q0; ++r0)
  //     {
  //       int i0 = idx(p0, q0, r0);
  //       for (int p1 = 0; p1 < n1 + 1; ++p1)
  //         for (int q1 = 0; q1 < n1 + 1 - p1; ++q1)
  //           for (int r1 = 0; r1 < n1 + 1 - p1 - q1; ++r1)
  //           {
  //             int i1 = idx(p1, q1, r1);
  //             int i2 = idx(p0 + p1, q0 + q1, r0 + r1);
  //             result.coeffs[i2] += this->coeffs[i0] * other.coeffs[i1];
  //           }
  //     }
  // return result;
};
