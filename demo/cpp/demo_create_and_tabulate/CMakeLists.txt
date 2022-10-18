cmake_minimum_required(VERSION 3.16)
project(demo_create_and_tabulate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (BASIX_PY_DIR)
  message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
endif()
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} Basix::basix)
