# Example of CMakeLists.txt for downstream C++ project (e.g. DOLFINx)
# Used to check correctly installed package during CI
cmake_minimum_required(VERSION 3.16)

project(basix_test)

# Use Python for detecting UFC and Basix
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# From DOLFINx
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (BASIX_PY_DIR)
  message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
endif()
find_package(xtl REQUIRED CONFIG HINTS ${BASIX_PY_DIR}/share/cmake/xtl)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

add_executable(a.out main.cpp)
target_link_libraries(a.out PRIVATE Basix::basix)
