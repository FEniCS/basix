// Copyright (c) 2026 Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <array>
#include <basix/sobolev-spaces.h>
#include <catch2/catch_test_macros.hpp>

using namespace basix;

TEST_CASE("Sobolev space intersection", "[sobolev]")
{
  constexpr std::array spaces
      = {sobolev::space::L2,    sobolev::space::H1,   sobolev::space::H2,
         sobolev::space::H3,    sobolev::space::HInf, sobolev::space::HDiv,
         sobolev::space::HCurl, sobolev::space::HEin, sobolev::space::HDivDiv};

  SECTION("A space intersected with itself is unchanged")
  {
    for (auto s : spaces)
      CHECK(sobolev::space_intersection(s, s) == s);
  }

  SECTION("The intersection is symmetric")
  {
    for (auto s0 : spaces)
    {
      for (auto s1 : spaces)
      {
        CHECK(sobolev::space_intersection(s0, s1)
              == sobolev::space_intersection(s1, s0));
      }
    }
  }

  SECTION("The H^k spaces are nested")
  {
    CHECK(sobolev::space_intersection(sobolev::space::H1, sobolev::space::H2)
          == sobolev::space::H1);
    CHECK(sobolev::space_intersection(sobolev::space::H2, sobolev::space::H3)
          == sobolev::space::H2);
    CHECK(sobolev::space_intersection(sobolev::space::H1, sobolev::space::HInf)
          == sobolev::space::H1);
    CHECK(sobolev::space_intersection(sobolev::space::H3, sobolev::space::HInf)
          == sobolev::space::H3);
  }

  SECTION("Unrelated spaces intersect in L2")
  {
    CHECK(
        sobolev::space_intersection(sobolev::space::HDiv, sobolev::space::HCurl)
        == sobolev::space::L2);
    CHECK(sobolev::space_intersection(sobolev::space::H1, sobolev::space::HDiv)
          == sobolev::space::L2);
    CHECK(sobolev::space_intersection(sobolev::space::HEin,
                                      sobolev::space::HDivDiv)
          == sobolev::space::L2);
    for (auto s : spaces)
    {
      if (s != sobolev::space::L2)
      {
        CHECK(sobolev::space_intersection(sobolev::space::L2, s)
              == sobolev::space::L2);
      }
    }
  }
}
