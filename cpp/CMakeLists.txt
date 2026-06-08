cmake_minimum_required(VERSION 3.21)

# Set the version
project(Basix VERSION "0.11.0.0" LANGUAGES CXX)
include(GNUInstallDirs)

if(SKBUILD AND UNIX)
  # Avoid lib64/ which can lead to linking issues.
  set(CMAKE_INSTALL_LIBDIR basix/lib)
endif()

if (SKBUILD)
  # Make sure includes end up inside basix/ subdirectory in wheel. 
  set(CMAKE_INSTALL_INCLUDEDIR basix/include)
  # Make sure all binaries end up inside basix/ subdirectory in wheel.
  set(CMAKE_INSTALL_BINDIR basix)
endif()

include(FeatureSummary)

# Options
option(BUILD_SHARED_LIBS "Build Basix with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build Basix with shared libraries.")

option(INSTALL_RUNTIME_DEPENDENCIES "Include runtime dependencies in install (Windows-only)" OFF)
add_feature_info(INSTALL_RUNTIME_DEPENDENCIES INSTALL_RUNTIME_DEPENDENCIES "Include runtime dependencies in install (Windows-only)")

option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(ENABLE_CLANG_TIDY ENABLE_CLANG_TIDY "Run clang-tidy while building")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

feature_summary(WHAT ALL)

if (WIN32)
    # Windows requires all symbols to be manually exported.
    # This flag exports all symbols automatically, as in Unix.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# Source files
add_library(basix)

# Set the C++ standard
target_compile_features(basix PUBLIC cxx_std_20)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/basix/version.h.in basix/version.h)
target_include_directories(basix PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/types.h
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

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(basix PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../.clang-tidy")
endif()

target_include_directories(basix PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}>")

target_compile_definitions(basix PUBLIC MDSPAN_USE_PAREN_OPERATOR=1)
target_compile_definitions(basix PUBLIC MDSPAN_USE_BRACKET_OPERATOR=0)

target_link_libraries(basix PRIVATE BLAS::BLAS)
target_link_libraries(basix PRIVATE LAPACK::LAPACK)

# Check for some compiler flags
include(CheckCXXCompilerFlag)

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic -Wno-comment" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  list(APPEND BASIX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic;-Wno-comment)
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)
if(HAVE_DEBUG)
  list(APPEND BASIX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)
if(HAVE_O2_OPTIMISATION)
  list(APPEND BASIX_CXX_DEVELOPER_FLAGS -O2)
endif()

# Pipe
check_cxx_compiler_flag(-pipe HAVE_PIPE)
if(HAVE_PIPE)
  list(APPEND BASIX_CXX_DEVELOPER_FLAGS -pipe)
endif()

# Enable C++ standard library debugging
include(CheckCXXSymbolExists)

check_cxx_symbol_exists(_LIBCPP_VERSION "version" LIBCPP)
check_cxx_symbol_exists(__GLIBCXX__ "version" GLIBCXX)

if(LIBCPP)
  list(APPEND BASIX_CXX_DEVELOPER_DEFINITIONS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
endif()

if(GLIBCXX)
  list(APPEND BASIX_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

target_compile_options(basix PRIVATE $<$<CONFIG:Developer>:${BASIX_CXX_DEVELOPER_FLAGS}>)
target_compile_definitions(basix PRIVATE $<$<CONFIG:Developer>:${BASIX_CXX_DEVELOPER_DEFINITIONS}>) 

if (MSVC)
    # M_PI etc. are not not in the standard-compliant MSVC headers.
    target_compile_definitions(basix PUBLIC _USE_MATH_DEFINES)
endif()

install(TARGETS basix
  EXPORT BasixTargets
  RUNTIME_DEPENDENCY_SET dependencies
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/basix
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
)
if (INSTALL_RUNTIME_DEPENDENCIES AND WIN32)
  # https://discourse.cmake.org/t/migration-experiences-comparison-runtime-dependency-set-vs-fixup-bundle-bundleutilities
  install(RUNTIME_DEPENDENCY_SET dependencies DESTINATION ${CMAKE_INSTALL_BINDIR} PRE_EXCLUDE_REGEXES [[api-ms-win-.*]] [[ext-ms-.*]] POST_EXCLUDE_REGEXES [[.*(\\|/)system32(\\|/).*\.dll]])
endif()

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
