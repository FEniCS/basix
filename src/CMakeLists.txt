# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -Wall -Werror -std=c++17" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -Wall -Werror -std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

# Executable
add_library(tab SHARED polynomial-set.cpp integral-moments.cpp lagrange.cpp quadrature.cpp nedelec.cpp nedelec-second-kind.cpp raviart-thomas.cpp finite-element.cpp tp.cpp cell.cpp regge.cpp crouzeix-raviart.cpp)
