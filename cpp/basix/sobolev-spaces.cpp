// Copyright (c) 2022 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "sobolev-spaces.h"

basix::sobolev::space
basix::sobolev::space_intersection(basix::sobolev::space space1,
                                   basix::sobolev::space space2)
{
  if (space1 == space2)
    return space1;
  if (space1 == basix::sobolev::space::H1)
  {
    if (space2 == basix::sobolev::space::H2
        || space2 == basix::sobolev::space::H3
        || space2 == basix::sobolev::space::HInf)
      return space1;
  }
  else if (space1 == basix::sobolev::space::H2)
  {
    if (space2 == basix::sobolev::space::H1)
      return space2;
    else if (space2 == basix::sobolev::space::H3
             || space2 == basix::sobolev::space::HInf)
      return space1;
  }
  else if (space1 == basix::sobolev::space::H3)
  {
    if (space2 == basix::sobolev::space::H1
        || space2 == basix::sobolev::space::H2)
      return space2;
    else if (space2 == basix::sobolev::space::H3
             || space2 == basix::sobolev::space::HInf)
      return space1;
  }
  else if (space1 == basix::sobolev::space::HInf)
  {
    if (space2 == basix::sobolev::space::H1
        || space2 == basix::sobolev::space::H2
        || space2 == basix::sobolev::space::H3)
      return space2;
  }

  return basix::sobolev::space::L2;
}
