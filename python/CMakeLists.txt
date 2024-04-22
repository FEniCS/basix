cmake_minimum_required(VERSION 3.19)

# nanobind uses aligned deallocators only present on macOS > 10.14
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
endif()

project(basix_nanobind VERSION "0.8.0" LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/16414
if(TARGET basix)
  add_library(Basix::basix ALIAS basix)
else()
  find_package(Basix REQUIRED)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Create the binding library
nanobind_add_module(_basixcpp wrapper.cpp)
target_link_libraries(_basixcpp PRIVATE Basix::basix)

# Add strict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -Wextra -Wno-comment -pedantic" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  target_compile_options(_basixcpp PRIVATE -Wall;-Wextra;-Werror;-Wno-comment)

  # target_compile_options(_basixcpp PRIVATE -Wall;-Wextra;-Werror;-Wno-comment;-pedantic)
endif()

# Add relative rpath so _basixcpp can find the Basix::basix library
# when the build is relocated
if(BASIX_FULL_SKBUILD)
  if(APPLE)
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "@loader_path/lib")
  else()
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
  endif()
else()
  get_target_property(_location Basix::basix LOCATION)
  get_filename_component(_basix_dir ${_location} DIRECTORY)
  set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH ${_basix_dir})
endif()

install(TARGETS _basixcpp LIBRARY DESTINATION basix)
