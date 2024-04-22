cmake_minimum_required(VERSION 3.19)

# Set the version
project(Basix VERSION "0.8.0" LANGUAGES CXX)
include(GNUInstallDirs)

if(SKBUILD)
  # Always use lib/ in the Python root. Otherwise, environment used for
  # wheels might place in lib64/, which CMake on some systems will not
  # find.
  set(CMAKE_INSTALL_LIBDIR basix/lib)
  set(CMAKE_INSTALL_INCLUDEDIR basix/include)
endif()

include(FeatureSummary)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build Basix with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build Basix with shared libraries.")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

feature_summary(WHAT ALL)

# Source files
add_library(basix)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/basix/version.h.in basix/version.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(HEADERS_basix
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/dof-transformations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/indexing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/interpolation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/lattice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/maps.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/polynomials.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/precompute.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/sobolev-spaces.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-lagrange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nce-rtc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-brezzi-douglas-marini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nedelec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-raviart-thomas.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-regge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-hhj.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-crouzeix-raviart.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-bubble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-serendipity.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-hermite.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/mdspan.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/basix/version.h)

target_sources(basix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/cell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/dof-transformations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/interpolation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/polynomials.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/precompute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/quadrature.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/sobolev-spaces.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-lagrange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nce-rtc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-brezzi-douglas-marini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nedelec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-raviart-thomas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-regge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-hhj.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-crouzeix-raviart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-bubble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-hermite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-serendipity.cpp)

# Configure the library
set_target_properties(basix PROPERTIES PUBLIC_HEADER basix/finite-element.h)
set_target_properties(basix PROPERTIES PRIVATE_HEADER "${HEADERS_basix}")
target_include_directories(basix PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}>")

target_link_libraries(basix PRIVATE BLAS::BLAS)
target_link_libraries(basix PRIVATE LAPACK::LAPACK)

# Set compiler flags
list(APPEND BASIX_DEVELOPER_FLAGS -O2;-g;-pipe)
list(APPEND basix_compiler_flags -Wall;-Werror;-Wextra;-Wno-comment;-pedantic;)
target_compile_options(basix PRIVATE "$<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:${basix_compiler_flags}>")
target_compile_options(basix PRIVATE $<$<CONFIG:Developer>:${BASIX_DEVELOPER_FLAGS}>)

# Install the Basix library
install(TARGETS basix
  EXPORT BasixTargets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/basix
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development)

# Configure CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(BasixConfigVersion.cmake VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(BasixConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/BasixConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)

# Install CMake files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/BasixConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/BasixConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix COMPONENT Development)
install(EXPORT BasixTargets FILE BasixTargets.cmake NAMESPACE Basix::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)
