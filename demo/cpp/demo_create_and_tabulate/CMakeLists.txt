cmake_minimum_required(VERSION 3.16)
project(demo_create_and_tabulate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Python for detecting Basix when installed using combined build
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/BasixPythonHints.cmake)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

add_executable(${PROJECT_NAME} main.cpp)

if(BASIX_PY_DIR AND IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES BUILD_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs
  )
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs
  )
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE Basix::basix)
