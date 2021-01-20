# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -Wall -Werror -std=c++17" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -Wall -Werror -std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

# Executable
add_library(basix SHARED lattice.cpp polyset.cpp dof-permutations.cpp
                          moments.cpp lagrange.cpp finite-element.cpp
                          log.cpp quadrature.cpp brezzi-douglas-marini.cpp
                          nedelec.cpp raviart-thomas.cpp cell.cpp nce-rtc.cpp
                          regge.cpp crouzeix-raviart.cpp basix.cpp)
target_compile_definitions(basix PRIVATE BASIX_VERSION=${PROJECT_VERSION})
target_compile_definitions(basix PUBLIC EIGEN_MAX_ALIGN_BYTES=32)
