cmake_minimum_required(VERSION 3.12)

project(scikit_build_example VERSION "0.0.1")

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Currently, Scikit-build does not support FindPython, so we convert the
# provided hints ourselves.
if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARY "${PYTHON_LIBRARY}")
endif()

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen ${EIGEN3_INCLUDE_DIRS}")
include_directories(${EIGEN3_INCLUDE_DIRS})

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src)

pybind11_add_module(libtab MODULE src/wrapper.cpp)

set_target_properties(libtab PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(libtab PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(libtab PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

if (APPLE)
  set_target_properties(libtab PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(libtab PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

target_link_libraries(libtab PRIVATE pybind11::module tab)

target_compile_definitions(libtab PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS tab libtab DESTINATION .)
