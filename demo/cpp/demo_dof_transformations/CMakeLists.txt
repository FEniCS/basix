cmake_minimum_required(VERSION 3.16)
project(demo_dof_transformations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Python for detecting Basix when installed using combined build
find_package(Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
    OUTPUT_VARIABLE BASIX_PY_DIR
    RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (BASIX_PY_DIR)
    message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
  endif()
endif()
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

add_executable(${PROJECT_NAME} main.cpp)
if (BASIX_PY_DIR AND IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
endif()
target_link_libraries(${PROJECT_NAME} Basix::basix)
