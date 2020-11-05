// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-permutations.h"

using namespace libtab;

Eigen::Array<int, Eigen::Dynamic, 1> dofperms::interval_reflection(int degree)
{
  if (degree <= 0)
    return {};
  Eigen::Array<int, Eigen::Dynamic, 1> perm(degree);
  for (int i = 0; i < degree; ++i)
    perm(i) = degree - 1 - i;
  return perm;
}

Eigen::Array<int, Eigen::Dynamic, 1> dofperms::triangle_reflection(int degree)
{
  if (degree <= 0)
    return {};

  const int n = degree * (degree + 1) / 2;
  Eigen::Array<int, Eigen::Dynamic, 1> perm(n);
  int p = 0;

  for (int st = 0; st < degree; ++st)
  {
    int dof = st;
    for (int add = degree; add > st; --add)
    {
      perm(p++) = dof;
      dof += add;
    }
  }
  return perm;
}

Eigen::Array<int, Eigen::Dynamic, 1> dofperms::triangle_rotation(int degree)
{
  if (degree <= 0)
    return {};

  const int n = degree * (degree + 1) / 2;
  Eigen::Array<int, Eigen::Dynamic, 1> perm(n);
  int p = 0;

  int st = n - 1;
  for (int i = 1; i <= degree; ++i)
  {
    int dof = st;
    for (int sub = i; sub <= degree; ++sub)
    {
      perm(p++) = dof;
      dof -= sub + 1;
    }
    st -= i;
  }
  return perm;
}
