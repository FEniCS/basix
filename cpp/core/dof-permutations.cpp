// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-permutations.h"

using namespace basix;

//-----------------------------------------------------------------------------
std::vector<int> dofperms::interval_reflection(int degree)
{
  std::vector<int> perm(degree);
  for (int i = 0; i < degree; ++i)
    perm[i] = degree - 1 - i;
  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> dofperms::triangle_reflection(int degree)
{
  const int n = degree * (degree + 1) / 2;
  std::vector<int> perm(n);
  int p = 0;

  for (int st = 0; st < degree; ++st)
  {
    int dof = st;
    for (int add = degree; add > st; --add)
    {
      perm[p++] = dof;
      dof += add;
    }
  }

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> dofperms::triangle_rotation(int degree)
{
  const int n = degree * (degree + 1) / 2;
  std::vector<int> perm(n);
  int p = 0;
  int st = n - 1;
  for (int i = 1; i <= degree; ++i)
  {
    int dof = st;
    for (int sub = i; sub <= degree; ++sub)
    {
      perm[p++] = dof;
      dof -= sub + 1;
    }
    st -= i;
  }

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> dofperms::quadrilateral_reflection(int degree)
{
  const int n = degree * degree;
  std::vector<int> perm(n);
  int p = 0;

  for (int st = 0; st < degree; ++st)
    for (int i = 0; i < degree; ++i)
      perm[p++] = st + i * degree;

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> dofperms::quadrilateral_rotation(int degree)
{
  const int n = degree * degree;
  std::vector<int> perm(n);
  int p = 0;

  for (int st = degree - 1; st >= 0; --st)
    for (int i = 0; i < degree; ++i)
      perm[st + degree * i] = p++;

  return perm;
}
//-----------------------------------------------------------------------------
ndarray<double, 2> dofperms::interval_reflection_tangent_directions(int degree)
{
  ndarray<double, 2> r(degree, degree, 0.0);
  for (int i = 0; i < degree; ++i)
    r(i, i) = -1;
  return r;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd dofperms::triangle_reflection_tangent_directions(int degree)
{
  Eigen::ArrayXXd dirs
      = Eigen::ArrayXXd::Zero(degree * (degree + 1), degree * (degree + 1));
  for (int i = 0; i < degree * (degree + 1); i += 2)
  {
    dirs(i, i + 1) = 1;
    dirs(i + 1, i) = 1;
  }

  return dirs;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd dofperms::triangle_rotation_tangent_directions(int degree)
{
  Eigen::ArrayXXd dirs
      = Eigen::ArrayXXd::Zero(degree * (degree + 1), degree * (degree + 1));
  for (int i = 0; i < degree * (degree + 1); i += 2)
  {
    dirs(i, i + 1) = -1;
    dirs(i + 1, i) = 1;
    dirs(i, i) = -1;
  }

  return dirs;
}
//-----------------------------------------------------------------------------
