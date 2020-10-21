# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME libtab)
project(${PROJECT_NAME})

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen ${EIGEN3_INCLUDE_DIRS}")
include_directories(${EIGEN3_INCLUDE_DIRS})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -Wall -std=c++17" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -Wall -std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

# Executable
add_library(tab SHARED polynomial-set.cpp lagrange.cpp quadrature.cpp nedelec.cpp raviart-thomas.cpp finite-element.cpp tp.cpp cell.cpp regge.cpp crouzeix-raviart.cpp)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

pybind11_add_module(libtab SHARED wrapper.cpp)
target_link_libraries(libtab PRIVATE pybind11::module tab)
