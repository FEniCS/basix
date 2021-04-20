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
set(XTENSOR_MIN_VERSION 0.23.4)
set(XTL_MIN_VERSION 0.7.0)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
  find_package(xtl ${XTL_MIN_VERSION}...<0.8 QUIET)
  find_package(xtensor ${XTENSOR_MIN_VERSION}...<0.24 QUIET)
else()
  find_package(xtl ${XTL_MIN_VERSION} QUIET)
  find_package(xtensor ${XTENSOR_MIN_VERSION} QUIET)
endif()
find_package(xtensor-blas 0.19 QUIET)
if(NOT xtl_FOUND OR NOT xtensor_FOUND OR NOT xtensor-blas_FOUND)
  include(FetchContent)
endif()
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
    GIT_TAG        0.23.4
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
    GIT_TAG        master
  )
  FetchContent_MakeAvailable(xtensor_blas)
else()
  message("found xtensor-blas ${xtensor-blas_VERSION}")
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

feature_summary(WHAT ALL)

# --Source files

add_library(basix)

set(HEADERS_basix
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/dof-transformations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/element-families.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/finite-element.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/indexing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/lattice.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/log.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/maps.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/moments.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/polyset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/precompute.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/quadrature.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-lagrange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-nce-rtc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-brezzi-douglas-marini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-nedelec.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-raviart-thomas.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-regge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-crouzeix-raviart.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-bubble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-serendipity.h)

target_sources(basix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/cell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/dof-transformations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/element-families.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/finite-element.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/lattice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/maps.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/moments.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/polyset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/precompute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/quadrature.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-lagrange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-nce-rtc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-brezzi-douglas-marini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-nedelec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-raviart-thomas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-regge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-crouzeix-raviart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-bubble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/basix/e-serendipity.cpp)

# Configure the library
set_target_properties(basix PROPERTIES PUBLIC_HEADER cpp/basix/finite-element.h)
set_target_properties(basix PROPERTIES PRIVATE_HEADER "${HEADERS_basix}")
target_include_directories(basix PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/cpp>")

# The commented lines need CMake >= 3.18
# target_link_libraries(basix PRIVATE BLAS::BLAS)
# target_link_libraries(basix PRIVATE LAPACK::LAPACK)
target_link_libraries(basix PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(basix PRIVATE ${LAPACK_LIBRARIES})

# Note: We would like to use `target_link_libraries`, but CMake does not
# apply the SYSTEM option to the includes, and this causes our strict
# compiler checks to fail on the xtensor headers.
# target_link_libraries(basix PRIVATE xtl)
# target_link_libraries(basix PRIVATE xtensor)
# target_link_libraries(basix PRIVATE xtensor-blas)
get_target_property(_xtl_include_dirs xtl INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(basix SYSTEM PRIVATE ${_xtl_include_dirs})
get_target_property(_xtensor_blas_include_dirs xtensor-blas INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(basix SYSTEM PRIVATE ${_xtensor_blas_include_dirs})
get_target_property(_xtensor_include_dirs xtensor INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(basix SYSTEM PRIVATE ${_xtensor_include_dirs})

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

# Install CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(BasixConfigVersion.cmake VERSION ${PACKAGE_VERSION} COMPATIBILITY AnyNewerVersion)
configure_package_config_file(BasixConfig.cmake.in ${CMAKE_BINARY_DIR}/BasixConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)

install(FILES ${CMAKE_BINARY_DIR}/BasixConfig.cmake ${CMAKE_BINARY_DIR}/BasixConfigVersion.cmake
        DESTINATION  ${CMAKE_INSTALL_LIBDIR}/cmake/basix COMPONENT Development)
install(EXPORT BasixTargets FILE BasixTargets.cmake NAMESPACE Basix:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/basix)
