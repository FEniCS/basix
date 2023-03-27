cmake_minimum_required(VERSION 3.16)

PROJECT(basix_nanobind VERSION "0.7.0.0"  LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/16414
if (TARGET basix)
  add_library(Basix::basix ALIAS basix)
else()
  # Find Basix (C++)
  find_package(Basix REQUIRED)
endif()

# Correct COMPONENT specification Development.Module added only in CMake 3.18
# and above.
if(NOT SKBUILD)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  else()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  endif()
endif()

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
#   $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(Python COMPONENTS Interpreter Development REQUIRED)


# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)


# Create the binding library
# pybind11_add_module(_basixcpp MODULE wrapper.cpp)
nanobind_add_module(_basixcpp MODULE wrapper.cpp)
target_link_libraries(_basixcpp PRIVATE Basix::basix)

# Add strict compiler flags
include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-Wall -Werror -Wextra -Wno-comment -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(_basixcpp PRIVATE -Wall;-Wextra;-Werror;-Wno-comment)
endif()

# In Debug mode override pybind11 symbols visibility. Symbols must be
# visible to backtrace_symbols() to produce nice logs
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(_basixcpp PRIVATE "-fvisibility=default")
endif()

# scikit-build specific steps
if (SKBUILD)
  # Add relative rpath so _basixcpp can find the Basix::basix library
  # when the build is relocated
  if (APPLE)
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "@loader_path/lib")
  else()
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
  endif()

  install(TARGETS _basixcpp DESTINATION .)
# install(TARGETS nanobind DESTINATION lib)
endif()
