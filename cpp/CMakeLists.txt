# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment -Wall -Werror -std=c++17" HAVE_NO_MULTLINE)
if (HAVE_NO_MULTLINE)
  set(CMAKE_CXX_FLAGS "-Wno-comment -Wall -Werror -std=c++17 ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(core)
add_subdirectory(elements)
add_library(basix SHARED basix.cpp ${CPP_elements} ${CPP_core})
target_compile_definitions(basix PRIVATE BASIX_VERSION=${PROJECT_VERSION})
target_compile_definitions(basix PUBLIC EIGEN_MAX_ALIGN_BYTES=32)
