cmake_minimum_required(VERSION 3.16)

# Set the version
project(Basix VERSION "0.0.1" LANGUAGES CXX)
include(GNUInstallDirs)

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

set(HEADERS_basix
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/dof-transformations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/indexing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lattice.h
  # ${CMAKE_CURRENT_SOURCE_DIR}/cpp/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/mappings.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/span.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lagrange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/nce-rtc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/brezzi-douglas-marini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/nedelec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/raviart-thomas.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/regge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/crouzeix-raviart.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bubble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serendipity.h)

set(CPP_basix
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/dof-transformations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/element-families.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/mappings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/quadrature.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lagrange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/nce-rtc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/brezzi-douglas-marini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/nedelec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/raviart-thomas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/regge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/crouzeix-raviart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bubble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/serendipity.cpp)

# Configure the library
add_library(basix ${CPP_basix} ${CPP_elements} cpp/basix.cpp)
# set_target_properties(basix PROPERTIES PUBLIC_HEADER cpp/basix.h)
# set_target_properties(basix PROPERTIES PRIVATE_HEADER "${HEADERS_basix}")
# set_target_properties(basix PROPERTIES PUBLIC_HEADER cpp/basix.h)
set_target_properties(basix PROPERTIES PUBLIC_HEADER "${HEADERS_basix}")
target_include_directories(basix PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/cpp>")

# Add dependencies to target
target_include_directories(basix SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(basix PUBLIC xtensor)

# Note: xtl needs special handling to apply SYSTEM to the includes to
# allow strict compiler checking
get_target_property(_xtl_include_dirs xtl INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(basix SYSTEM PRIVATE ${_xtl_include_dirs})

# Set debug definitions (private)
list(APPEND basix_compiler_flags "-Wall -Werror -Wextra -pedantic")
target_compile_options(basix PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:${basix_compiler_flags}>)
target_compile_definitions(basix PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:DEBUG>)

# Install the Basix library
install(TARGETS basix
  EXPORT BasixTargets
  # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  # PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/basix
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/basix
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development)

install(FILES cpp/basix.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(BasixConfigVersion.cmake VERSION ${PACKAGE_VERSION} COMPATIBILITY AnyNewerVersion)
configure_package_config_file(BasixConfig.cmake.in ${CMAKE_BINARY_DIR}/BasixConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)

install(FILES ${CMAKE_BINARY_DIR}/BasixConfig.cmake ${CMAKE_BINARY_DIR}/BasixConfigVersion.cmake
        DESTINATION  ${CMAKE_INSTALL_LIBDIR}/cmake/basix COMPONENT Development)
install(EXPORT BasixTargets FILE BasixTargets.cmake NAMESPACE Basix:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)