# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -Wall -Werror -std=c++17" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -Wall -Werror -std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

# Executable
add_library(tab SHARED lattice.cpp defines.cpp polynomial-set.cpp dof-permutations.cpp integral-moments.cpp lagrange.cpp libtab.cpp quadrature.cpp
                       nedelec.cpp raviart-thomas.cpp cell.cpp regge.cpp crouzeix-raviart.cpp)
target_compile_definitions(tab PRIVATE LIBTAB_VERSION=${PROJECT_VERSION})
