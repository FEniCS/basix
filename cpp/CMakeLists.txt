# Copyright (c) 2020 Chris Richardson
# FEniCS Project
# SPDX-License-Identifier: MIT

# Note: -Wno-comment is added to avoid warning/errors for LaTeX code
# with backslashes on Doxygen comments
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -Wextra -pedantic -Wno-comment" HAVE_EXTRA_FLAGS)
if (HAVE_EXTRA_FLAGS)
  set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -pedantic -Wno-comment -DDEBUG ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(core)
add_subdirectory(elements)
add_library(basix SHARED basix.cpp ${CPP_elements} ${CPP_core})
target_compile_definitions(basix PRIVATE BASIX_VERSION=${PROJECT_VERSION})
target_compile_definitions(basix PUBLIC EIGEN_MAX_ALIGN_BYTES=32)
