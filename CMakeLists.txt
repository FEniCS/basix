cmake_minimum_required(VERSION 3.16)

# Set the version
project(Basix VERSION "0.0.1" LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# TODO: Replace hardwiring
# Configure compiler
# Note: -Wno-comment is added to avoid warning/errors for LaTeX code
# with backslashes on Doxygen comments
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -Wno-comment" HAVE_EXTRA_FLAGS)
if (HAVE_EXTRA_FLAGS)
  set(CMAKE_CXX_FLAGS "-Wall -Werror -Wno-comment ${CMAKE_CXX_FLAGS}")
endif()

# Options
include(FeatureSummary)
option(BUILD_SHARED_LIBS "Build Basix with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build Basix with shared libraries.")

# Find dependecies

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen ${EIGEN3_INCLUDE_DIRS}")

# find_package(xtl REQUIRED)
# find_package(xtensor REQUIRED)
include(FetchContent)
FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG        0.7.2
)
FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG        0.23.1
)
FetchContent_MakeAvailable(xtl)
FetchContent_MakeAvailable(xtensor)

feature_summary(WHAT ALL)

# --Source files

set(HEADERS_core
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/dof-transformations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/indexing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/lattice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/mappings.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/ndarray.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/span.hpp)

set(CPP_core
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/cell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/dof-transformations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/element-families.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/mappings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/quadrature.cpp)

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
set_target_properties(basix PROPERTIES PUBLIC_HEADER cpp/basix.h)
set_target_properties(basix PROPERTIES PRIVATE_HEADER "${HEADERS_core};${HEADERS_elements}")
target_include_directories(basix PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/cpp>")

# Add dependencies to target
target_include_directories(basix SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIRS})
# target_include_directories(basix SYSTEM PRIVATE ${xtl_INCLUDE_DIRS})
# target_include_directories(basix SYSTEM PUBLIC ${xtensor_INCLUDE_DIRS})
# target_link_libraries(basix SYSTEM PRIVATE xtl)

# get_property(_xtensor_include_dirs TARGET xtensor PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
# set_property(TARGET xtensor PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_xtensor_include_dirs}")

target_link_libraries(basix PUBLIC xtl)
# target_include_directories(basix SYSTEM PUBLIC ${xlt_INCLUDE_DIRS})
target_link_libraries(basix PUBLIC xtensor)


# # Temp fix
# target_compile_definitions(basix PUBLIC NDEBUG)


# Install the Basix library
install(TARGETS basix
  EXPORT BasixTargets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/basix
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development)

# Install CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(BasixConfigVersion.cmake VERSION ${PACKAGE_VERSION} COMPATIBILITY AnyNewerVersion)
configure_package_config_file(BasixConfig.cmake.in ${CMAKE_BINARY_DIR}/BasixConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)

install(FILES ${CMAKE_BINARY_DIR}/BasixConfig.cmake ${CMAKE_BINARY_DIR}/BasixConfigVersion.cmake
        DESTINATION  ${CMAKE_INSTALL_LIBDIR}/cmake/basix COMPONENT Development)

install(EXPORT BasixTargets FILE BasixTargets.cmake NAMESPACE Basix:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)