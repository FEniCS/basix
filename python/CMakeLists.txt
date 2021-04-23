cmake_minimum_required(VERSION 3.16)

PROJECT(basix_pybind11 VERSION "0.0.1")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Basix (C++)
find_package(Basix REQUIRED)

# Check if the Basix C++ library was compiled with xtensor SIMD
get_target_property(BASIX_DEFN Basix::basix INTERFACE_COMPILE_DEFINITIONS)
if("XTENSOR_USE_XSIMD" IN_LIST BASIX_DEFN)
  find_package(xsimd REQUIRED)
endif()

# include(FetchContent)
# FetchContent_Declare(
#   basix
#   GIT_REPOSITORY https://github.com/FEniCS/basix.git
#   GIT_TAG        xtensor
# )
# FetchContent_MakeAvailable(basix)
# get_target_property(_basix_include_dirs basix INTERFACE_INCLUDE_DIRECTORIES)
# target_include_directories(basix SYSTEM PRIVATE ${_basix_include_dirs})

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

# Create the binding library
pybind11_add_module(_basixcpp SHARED wrapper.cpp)
target_link_libraries(_basixcpp PRIVATE pybind11::module)
target_link_libraries(_basixcpp PRIVATE Basix::basix)

# Add xsimd if the Basix C++ interface was compiled with it
if("XTENSOR_USE_XSIMD" IN_LIST BASIX_DEFN)
  target_link_libraries(_basixcpp PRIVATE xsimd)
endif()

# Add strict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror  -Wextra -Wno-comment -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(_basixcpp PRIVATE -Wall;-Wextra;-Werror;-Wno-comment;-pedantic)
endif()

# In Debug mode override pybind11 symbols visibility Symbols must be
# visible to backtrace_symbols() to produce nice logs
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(_basixcpp PRIVATE "-fvisibility=default")
endif()


