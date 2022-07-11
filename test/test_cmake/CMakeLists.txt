# Example of CMakeLists.txt for downstream C++ project (eg DOLFINx)
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
endif()
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

add_executable(a.out main.cpp)
if (BASIX_PY_DIR AND IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
    set_target_properties(a.out PROPERTIES BUILD_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
    set_target_properties(a.out PROPERTIES INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
endif()
target_link_libraries(a.out PRIVATE Basix::basix)
