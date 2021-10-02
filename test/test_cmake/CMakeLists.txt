# Example of CMakeLists.txt for downstream C++ project (e.g. DOLFINx)
# Used to check correctly installed package during wheel creation.:
cmake_minimum_required(VERSION 3.16)

project(basix_test)

# Use Python for detecting Basix
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# From DOLFINx
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (BASIX_PY_DIR)
  message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
  # Lines below are when using pip-installed Basix
  set(xtl_ROOT "${BASIX_PY_DIR};${xtl_ROOT}")
  set(xsimd "${BASIX_PY_DIR};${xsimd_ROOT}")
  set(xtensor_ROOT "${BASIX_PY_DIR};${xtensor_ROOT}")
  # Lines below are necessary when using binary wheel Basix
  set(CMAKE_BUILD_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
  set(CMAKE_INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
endif()
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

get_target_property(BASIX_DEFN Basix::basix INTERFACE_COMPILE_DEFINITIONS)
if("XTENSOR_USE_XSIMD" IN_LIST BASIX_DEFN)
  find_package(xsimd REQUIRED)
endif()

add_executable(a.out main.cpp)
target_link_libraries(a.out PRIVATE Basix::basix)
