// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "dof-transformations.h"

using namespace basix;

//-----------------------------------------------------------------------------
std::vector<int> doftransforms::interval_reflection(int degree)
{
  std::vector<int> perm(std::max(0, degree));
  for (int i = 0; i < degree; ++i)
    perm[i] = degree - 1 - i;
  return perm;
}
//-----------------------------------------------------------------------------
std::vector<int> doftransforms::triangle_reflection(int degree)
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
std::vector<int> doftransforms::triangle_rotation(int degree)
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
std::vector<int> doftransforms::quadrilateral_reflection(int degree)
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
std::vector<int> doftransforms::quadrilateral_rotation(int degree)
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
