cmake_minimum_required(VERSION 3.21)

# nanobind uses aligned deallocators only present on macOS > 10.14
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
endif()

project(basix_nanobind VERSION "0.9.0.0" LANGUAGES CXX)

if (WIN32)
    # Windows requires all symbols to be manually exported.
    # This flag exports all symbols automatically, as in Unix.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/16414
if(TARGET basix)
  add_library(Basix::basix ALIAS basix)
else()
  find_package(Basix REQUIRED)
endif()

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# Options
include(FeatureSummary)

option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(ENABLE_CLANG_TIDY ENABLE_CLANG_TIDY "Run clang-tidy while building")

feature_summary(WHAT ALL)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Create the binding library
if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_SABI "STABLE_ABI")
endif()
nanobind_add_module(_basixcpp NB_SUPPRESS_WARNINGS ${NANOBIND_SABI} wrapper.cpp)
target_compile_definitions(_basixcpp PRIVATE cxx_std_20)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(_basixcpp PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../.clang-tidy")
endif()

target_link_libraries(_basixcpp PRIVATE Basix::basix)

# Add strict compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)
if(HAVE_PEDANTIC)
  target_compile_options(_basixcpp PRIVATE -Wall;-Werror;-Wextra;-Wno-comment;-pedantic)
  # Intel's complex header uses #include_next which is not allowed with -Wall
  if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(_basixcpp PRIVATE -Wno-gnu-include-next)
  endif()
endif()

# TODO: remove after issue https://github.com/FEniCS/basix/issues/842 is resolved
target_compile_definitions(_basixcpp PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
target_compile_definitions(_basixcpp PRIVATE MDSPAN_USE_BRACKET_OPERATOR=0)

# Add relative rpath so _basixcpp can find the Basix::basix library
# when the build is relocated
if(BASIX_FULL_SKBUILD)
  if(APPLE)
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "@loader_path/lib")
  elseif(UNIX)
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
  endif()
else()
  get_target_property(_location Basix::basix LOCATION)
  get_filename_component(_basix_dir ${_location} DIRECTORY)
  set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH ${_basix_dir})
endif()

install(TARGETS _basixcpp LIBRARY DESTINATION basix)
