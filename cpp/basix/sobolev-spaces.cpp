// Copyright (c) 2022 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "sobolev-spaces.h"

basix::sobolev::space basix::sobolev::space_intersection(space space1,
                                                         space space2)
{
  if (space1 == space2)
    return space1;

  if (space1 == space::H1)
  {
    if (space2 == space::H2 or space2 == space::H3 or space2 == space::HInf)
      return space1;
  }
  else if (space1 == space::H2)
  {
    if (space2 == space::H1)
      return space2;
    else if (space2 == space::H3 or space2 == space::HInf)
      return space1;
  }
  else if (space1 == space::H3)
  {
    if (space2 == space::H1 or space2 == space::H2)
      return space2;
    else if (space2 == space::H3 or space2 == space::HInf)
      return space1;
  }
  else if (space1 == space::HInf)
  {
    if (space2 == space::H1 or space2 == space::H2 or space2 == space::H3)
      return space2;
  }

  return space::L2;
}
