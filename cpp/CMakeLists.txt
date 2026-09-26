cmake_minimum_required(VERSION 3.21)

# Set the version
project(
  Basix
  VERSION 0.12.0.0
  DESCRIPTION "Finite element definition and tabulation runtime library"
  HOMEPAGE_URL "https://fenicsproject.org"
  LANGUAGES CXX
)
include(GNUInstallDirs)

if(SKBUILD)
  # Make sure libraries, the CMake package and basix.pc end up inside basix/
  # subdirectory in wheel. Also avoids lib64/, which can lead to linking issues.
  set(CMAKE_INSTALL_LIBDIR basix/lib)
  # Make sure includes end up inside basix/ subdirectory in wheel.
  set(CMAKE_INSTALL_INCLUDEDIR basix/include)
  # Make sure all binaries end up inside basix/ subdirectory in wheel.
  set(CMAKE_INSTALL_BINDIR basix)
endif()

include(FeatureSummary)

# Options
option(BUILD_SHARED_LIBS "Build Basix with shared libraries." ON)
add_feature_info(
  BUILD_SHARED_LIBS
  BUILD_SHARED_LIBS
  "Build Basix with shared libraries."
)

option(
  INSTALL_RUNTIME_DEPENDENCIES
  "Include runtime dependencies in install (Windows-only)"
  OFF
)
add_feature_info(
  INSTALL_RUNTIME_DEPENDENCIES
  INSTALL_RUNTIME_DEPENDENCIES
  "Include runtime dependencies in install (Windows-only)"
)

option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(
  ENABLE_CLANG_TIDY
  ENABLE_CLANG_TIDY
  "Run clang-tidy while building"
)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

feature_summary(WHAT ALL)

# Source files
add_library(basix)

# Windows requires all symbols to be manually exported; this exports them
# automatically as on Unix. No-op on other platforms.
set_target_properties(basix PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# Set the C++ standard
target_compile_features(basix PUBLIC cxx_std_20)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/basix/version.h.in basix/version.h)

set(
  HEADERS_basix
  basix/cell.h
  basix/dof-transformations.h
  basix/element-families.h
  basix/finite-element.h
  basix/indexing.h
  basix/interpolation.h
  basix/lattice.h
  basix/maps.h
  basix/math.h
  basix/moments.h
  basix/polynomials.h
  basix/polyset.h
  basix/precompute.h
  basix/quadrature.h
  basix/sobolev-spaces.h
  basix/e-lagrange.h
  basix/e-nce-rtc.h
  basix/e-brezzi-douglas-marini.h
  basix/e-nedelec.h
  basix/e-raviart-thomas.h
  basix/e-regge.h
  basix/e-hhj.h
  basix/e-crouzeix-raviart.h
  basix/e-bubble.h
  basix/e-serendipity.h
  basix/e-hermite.h
  basix/mdspan.hpp
  basix/types.h
  ${CMAKE_CURRENT_BINARY_DIR}/basix/version.h
)

target_sources(
  basix
  PRIVATE
    basix/cell.cpp
    basix/dof-transformations.cpp
    basix/finite-element.cpp
    basix/interpolation.cpp
    basix/lattice.cpp
    basix/moments.cpp
    basix/polynomials.cpp
    basix/polyset.cpp
    basix/precompute.cpp
    basix/quadrature.cpp
    basix/sobolev-spaces.cpp
    basix/e-lagrange.cpp
    basix/e-nce-rtc.cpp
    basix/e-brezzi-douglas-marini.cpp
    basix/e-nedelec.cpp
    basix/e-raviart-thomas.cpp
    basix/e-regge.cpp
    basix/e-hhj.cpp
    basix/e-crouzeix-raviart.cpp
    basix/e-bubble.cpp
    basix/e-hermite.cpp
    basix/e-serendipity.cpp
)

# Configure the library
set_target_properties(
  basix
  PROPERTIES
    PUBLIC_HEADER basix/finite-element.h
    PRIVATE_HEADER "${HEADERS_basix}"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    basix
    PROPERTIES
      CXX_CLANG_TIDY
        "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../.clang-tidy"
  )
endif()

target_include_directories(
  basix
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}>"
)

target_compile_definitions(
  basix
  PUBLIC MDSPAN_USE_PAREN_OPERATOR=1 MDSPAN_USE_BRACKET_OPERATOR=0
)

target_link_libraries(basix PRIVATE BLAS::BLAS LAPACK::LAPACK)

# Check for some compiler flags
include(CheckCXXCompilerFlag)

set(BASIX_CXX_DEVELOPER_FLAGS "")
set(BASIX_CXX_DEVELOPER_DEFINITIONS "")

# Add some strict compiler checks
check_cxx_compiler_flag(
  "-Wall -Werror -Wextra -pedantic -Wno-comment"
  HAVE_PEDANTIC
)

if(HAVE_PEDANTIC)
  list(
    APPEND BASIX_CXX_DEVELOPER_FLAGS
    -Wall;-Werror;-Wextra;-pedantic;-Wno-comment
  )
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
  list(
    APPEND BASIX_CXX_DEVELOPER_DEFINITIONS
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
  )
endif()

if(GLIBCXX)
  list(APPEND BASIX_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

target_compile_options(
  basix
  PRIVATE $<$<CONFIG:Developer>:${BASIX_CXX_DEVELOPER_FLAGS}>
)
target_compile_definitions(
  basix
  PRIVATE $<$<CONFIG:Developer>:${BASIX_CXX_DEVELOPER_DEFINITIONS}>
)

if(MSVC)
  # M_PI etc. are not not in the standard-compliant MSVC headers.
  target_compile_definitions(basix PUBLIC _USE_MATH_DEFINES)
endif()

install(
  TARGETS basix
  EXPORT BasixTargets
  RUNTIME_DEPENDENCY_SET dependencies
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/basix
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
)
if(INSTALL_RUNTIME_DEPENDENCIES AND WIN32)
  # Regexes below accept either separator, so CMP0207 normalisation is a no-op.
  if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
  endif()
  # https://discourse.cmake.org/t/migration-experiences-comparison-runtime-dependency-set-vs-fixup-bundle-bundleutilities
  install(
    RUNTIME_DEPENDENCY_SET dependencies DESTINATION ${CMAKE_INSTALL_BINDIR}
    PRE_EXCLUDE_REGEXES [[api-ms-win-.*]] [[ext-ms-.*]]
    POST_EXCLUDE_REGEXES [[.*(\\|/)system32(\\|/).*\.dll]]
  )
endif()

# Configure CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  BasixConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  BasixConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BasixConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix
)

# Install CMake files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BasixConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BasixConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix
  COMPONENT Development
)
install(
  EXPORT BasixTargets
  FILE BasixTargets.cmake
  NAMESPACE Basix::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix
)

# Install a Common Package Specification (CPS) file. install(PACKAGE_INFO)
# is stable as of CMake 4.3.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.3")
  install(
    PACKAGE_INFO Basix
    EXPORT BasixTargets
    VERSION ${PROJECT_VERSION}
    DEFAULT_TARGETS basix
    DEFAULT_CONFIGURATIONS Release
    LICENSE MIT
    COMPONENT Development
  )
endif()

# Configure and install pkg-config file
get_target_property(PKG_BASIX_DEFINITIONS basix INTERFACE_COMPILE_DEFINITIONS)
set(PKG_DEFINITIONS)
foreach(_def ${PKG_BASIX_DEFINITIONS})
  set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
get_target_property(_cxx_standard basix CXX_STANDARD)
if(_cxx_standard)
  string(APPEND PKG_CXXFLAGS " -std=c++${_cxx_standard}")
endif()

# Only a static Basix::basix needs its consumers to also link
# BLAS/LAPACK, so they go in Libs.private.
# Resolving BLAS::BLAS/LAPACK::LAPACK's own IMPORTED_LOCATION, rather
# than the raw BLAS_LIBRARIES/LAPACK_LIBRARIES variables, keeps this in
# sync with what basix actually links.
# macOS frameworks aren't valid bare linker arguments, so translate them to -framework.
function(_basix_pkgconfig_append lib out_var)
  set(_result "${${out_var}}")
  if(lib MATCHES "\\.framework$")
    get_filename_component(_fw_name "${lib}" NAME_WE)
    list(APPEND _result "-framework" "${_fw_name}")
  else()
    list(APPEND _result "${lib}")
  endif()
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

function(_basix_pkgconfig_resolve_target target out_var)
  set(_result "${${out_var}}")
  string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" _target "${target}")
  if(NOT TARGET "${_target}")
    # Not a CMake target: already a plain linker flag/path.
    _basix_pkgconfig_append("${_target}" _result)
    set(${out_var} "${_result}" PARENT_SCOPE)
    return()
  endif()

  get_target_property(_type "${_target}" TYPE)
  set(_loc)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY")
    set(_candidates)
    if(CMAKE_BUILD_TYPE)
      string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfg_upper)
      list(APPEND _candidates "IMPORTED_LOCATION_${_cfg_upper}")
    endif()
    list(APPEND _candidates IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_NOCONFIG
         IMPORTED_LOCATION
    )
    foreach(_prop ${_candidates})
      get_target_property(_loc "${_target}" "${_prop}")
      if(_loc)
        break()
      endif()
    endforeach()
  endif()

  if(_loc)
    _basix_pkgconfig_append("${_loc}" _result)
  else()
    # Recurse into what an INTERFACE target links to expand it fully.
    get_target_property(_iface_libs "${_target}" INTERFACE_LINK_LIBRARIES)
    if(_iface_libs)
      foreach(_lib ${_iface_libs})
        _basix_pkgconfig_resolve_target("${_lib}" _result)
      endforeach()
    endif()
  endif()
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

set(_pkg_libs_private)
foreach(_target BLAS::BLAS LAPACK::LAPACK)
  _basix_pkgconfig_resolve_target("${_target}" _pkg_libs_private)
endforeach()
string(REPLACE ";" " " PKG_LIBS_PRIVATE "${_pkg_libs_private}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/basix.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/basix.pc
  @ONLY
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/basix.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT Development
)
