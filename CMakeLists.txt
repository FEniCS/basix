cmake_minimum_required(VERSION 3.18)

# Set the version
project(basix VERSION "0.0.1" LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Options
include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build Basix with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build Basix with shared libraries.")

# Find dependecies

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen ${EIGEN3_INCLUDE_DIRS}")

feature_summary(WHAT ALL)

# --Source files

set(HEADERS_core
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/lattice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/ndarray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/dof-transformations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/mappings.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/span.hpp)

set(CPP_core
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/dof-transformations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/mappings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/element-families.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/quadrature.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/cell.cpp)

  set(HEADERS_elements
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/lagrange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/nce-rtc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/brezzi-douglas-marini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/nedelec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/raviart-thomas.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/regge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/crouzeix-raviart.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/bubble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/serendipity.h)

set(CPP_elements
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/lagrange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/nce-rtc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/brezzi-douglas-marini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/nedelec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/raviart-thomas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/regge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/crouzeix-raviart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/bubble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/elements/serendipity.cpp)

# Configure the library
add_library(basix ${CPP_core} ${CPP_elements} cpp/basix.cpp)
target_include_directories(basix PRIVATE ./cpp)
target_include_directories(basix PRIVATE ${EIGEN3_INCLUDE_DIRS})
set_target_properties(basix PROPERTIES PUBLIC_HEADER cpp/basix.h)

# Install the Basix library
install(TARGETS basix
EXPORT BasixTargets
RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development)

# Install CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(BasixConfigVersion.cmake VERSION ${PACKAGE_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(EXPORT BasixTargets FILE BasixTargets.cmake NAMESPACE Basix:: DESTINATION lib/cmake/basix)


# target_link_libraries (helloDemo LINK_PUBLIC Hello)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
# include(GNUInstallDirs)

# # Currently, Scikit-build does not support FindPython, so we convert the
# # provided hints ourselves.
# if(SKBUILD)
#    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
#    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
#    set(Python_LIBRARY "${PYTHON_LIBRARY}")
# endif()

# set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
# find_package(Python REQUIRED COMPONENTS Interpreter Development)


# # Get pybind11 CMake dir
# execute_process(
#   COMMAND
#     "${Python_EXECUTABLE}" -c
#     "import pybind11; print(pybind11.get_cmake_dir())"
#   OUTPUT_VARIABLE _tmp_cmake_dir
#   OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
#   ECHO_ERROR_VARIABLE)
# list(APPEND CMAKE_PREFIX_PATH "${_tmp_cmake_dir}")

# find_package(Eigen3 REQUIRED)
# message(STATUS "Found Eigen ${EIGEN3_INCLUDE_DIRS}")
# include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

# # Now we can find pybind11
# find_package(pybind11 CONFIG REQUIRED)

# add_subdirectory(cpp)

# pybind11_add_module(_basixcpp MODULE cpp/wrapper.cpp)

# set_target_properties(_basixcpp PROPERTIES SKIP_BUILD_RPATH FALSE)
# set_target_properties(_basixcpp PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
# set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# if (APPLE)
#   set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "@loader_path/lib")
# else()
#   set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
# endif()

# target_link_libraries(_basixcpp PRIVATE pybind11::module basix)

# set_target_properties(basix PROPERTIES PUBLIC_HEADER "cpp/basix.h")

# install(TARGETS _basixcpp DESTINATION .)
# install(TARGETS basix LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
