cmake_minimum_required(VERSION 3.16)

# Set the version
project(Basix VERSION "0.3.1.0" LANGUAGES CXX)
include(GNUInstallDirs)

if (SKBUILD)
  # Always use lib/ in the Python root. Otherwise, environment used for
  # wheels might place in lib64/, which CMake on some systems will not
  # find.
  set(CMAKE_INSTALL_LIBDIR lib)
endif()

include(FeatureSummary)

# Enable SIMD with xtensor
option(XTENSOR_USE_XSIMD "Enable SIMD with xtensor" OFF)
add_feature_info(XTENSOR_USE_XSIMD XTENSOR_USE_XSIMD "Enable SIMD with xtensor.")

# Enable xtensor with target-specific optimization, i.e. -march=native
option(XTENSOR_OPTIMIZE "Enable xtensor target-specific optimization" OFF)
add_feature_info(XTENSOR_OPTIMIZE XTENSOR_OPTIMIZE "Enable architecture-specific optimizations as defined by xtensor.")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build Basix with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build Basix with shared libraries.")

# Find dependecies
set(XTENSOR_MIN_VERSION 0.23.10)
set(XTL_MIN_VERSION 0.7.0)
set(XSIMD_MIN_VERSION 7.4.10)

include(FetchContent)

# xsimd (optional)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
  find_package(xsimd ${XSIMD_MIN_VERSION}...<8.0 QUIET)
else()
  find_package(xsimd ${XSIMD_MIN_VERSION} QUIET)
endif()
if(XTENSOR_USE_XSIMD AND NOT xsimd_FOUND)
  message("downloading xsimd source...")
  FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        7.4.10
  )
  FetchContent_MakeAvailable(xsimd)
else()
  message("found xsimd ${xsimd_VERSION}")
endif()

# xtl, xtensor, xtensor-blas (required)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
  find_package(xtl ${XTL_MIN_VERSION}...<0.8 QUIET)
  find_package(xtensor ${XTENSOR_MIN_VERSION}...<0.24 QUIET)
else()
  find_package(xtl ${XTL_MIN_VERSION} QUIET)
  find_package(xtensor ${XTENSOR_MIN_VERSION} QUIET)
endif()
find_package(xtensor-blas 0.19 QUIET)

if(NOT xtl_FOUND)
  message("downloading xtl source...")
  FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        0.7.2
  )
  FetchContent_MakeAvailable(xtl)
else()
  message("found xtl ${xtl_VERSION}")
endif()

if(NOT xtensor_FOUND)
  message("downloading xtensor source...")
  FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        0.23.10
  )
  FetchContent_MakeAvailable(xtensor)
else()
  message("found xtensor ${xtensor_VERSION}")
endif()

if(NOT xtensor-blas_FOUND)
  message("downloading xtensor-blas source...")
  FetchContent_Declare(
    xtensor_blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
    GIT_TAG        0.19.1
  )
  # Not using FetchContent_MakeAvailable(xtensor_blas) because we don't
  # want xtensor_blas to be installed
  FetchContent_GetProperties(xtensor_blas)
  if(NOT xtensor_blas_POPULATED)
    FetchContent_Populate(xtensor_blas)
    add_subdirectory(${xtensor_blas_SOURCE_DIR} ${xtensor_blas_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
else()
  message("found xtensor-blas ${xtensor-blas_VERSION}")
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

feature_summary(WHAT ALL)

# --Source files

add_library(basix)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/basix/version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/basix/version.h)

set(HEADERS_basix
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/dof-transformations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/indexing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/lattice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/maps.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/precompute.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-lagrange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nce-rtc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-brezzi-douglas-marini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nedelec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-raviart-thomas.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-regge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-crouzeix-raviart.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-bubble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-serendipity.h
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/version.h)

target_sources(basix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/cell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/dof-transformations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/precompute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/quadrature.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-lagrange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nce-rtc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-brezzi-douglas-marini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-nedelec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-raviart-thomas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-regge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-crouzeix-raviart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-bubble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basix/e-serendipity.cpp)

# Configure the library
set_target_properties(basix PROPERTIES PUBLIC_HEADER basix/finite-element.h)
set_target_properties(basix PROPERTIES PRIVATE_HEADER "${HEADERS_basix}")
target_include_directories(basix PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}>")

# The commented lines need CMake >= 3.18
# target_link_libraries(basix PRIVATE BLAS::BLAS)
# target_link_libraries(basix PRIVATE LAPACK::LAPACK)
target_link_libraries(basix PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(basix PRIVATE ${LAPACK_LIBRARIES})

# xtensor and related packages
target_link_libraries(basix PUBLIC xtl)

# Note: we use get_target_property/set_target_properties to ensure that
# that -isystem flag is applied to allow us to use strict compiler flags
get_target_property(XTENSOR_INC_SYS xtensor INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(xtensor PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${XTENSOR_INC_SYS}")
target_link_libraries(basix PUBLIC xtensor)

# Add xsimd definition to Basix so that Basix will export this option so
# that other libraries will know that Basix was compiled with xsimd
get_target_property(XTENSOR_DEFN xtensor INTERFACE_COMPILE_DEFINITIONS)
if("XTENSOR_USE_XSIMD" IN_LIST XTENSOR_DEFN)
  target_compile_definitions(basix PUBLIC XTENSOR_USE_XSIMD)
elseif(TARGET xsimd AND XTENSOR_USE_XSIMD)
  target_compile_definitions(basix PUBLIC XTENSOR_USE_XSIMD)
  target_link_libraries(basix PRIVATE xsimd)
endif()

# Handle -march=native
if (XTENSOR_OPTIMIZE AND TARGET xtensor::optimize)
  target_link_libraries(basix PUBLIC xtensor::optimize)
elseif(XTENSOR_OPTIMIZE)
  target_compile_options(basix PUBLIC -march=native)
endif()

# Note: we use get_target_property/set_target_properties to ensure that
# that -isystem flag is applied to allow us to use strict compiler flags
get_target_property(XTENSORBLAS_INC_SYS xtensor-blas INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(xtensor-blas PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${XTENSORBLAS_INC_SYS}")
target_link_libraries(basix PRIVATE xtensor-blas)

# Set compiler flags
list(APPEND BASIX_DEVELOPER_FLAGS -O2;-g;-pipe)
list(APPEND basix_compiler_flags -Wall;-Werror;-Wextra;-Wno-comment;-pedantic;)
target_compile_options(basix PRIVATE "$<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:${basix_compiler_flags}>")
target_compile_options(basix PRIVATE $<$<CONFIG:Developer>:${BASIX_DEVELOPER_FLAGS}>)

# Set debug definitions (private)
target_compile_definitions(basix PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:DEBUG>)

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
  DESTINATION  ${CMAKE_INSTALL_LIBDIR}/cmake/basix COMPONENT Development)
install(EXPORT BasixTargets FILE BasixTargets.cmake NAMESPACE Basix::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)
