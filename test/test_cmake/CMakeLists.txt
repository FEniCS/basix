# Example of CMakeLists.txt for downstream C++ project (eg DOLFINx or a Basix demo)
# Used to check correctly installed package during tests and wheel creation.
cmake_minimum_required(VERSION 3.16)

project(basix_test)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Python for detecting Basix
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/BasixPythonHints.cmake)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

add_executable(a.out main.cpp)
if(BASIX_PY_DIR AND IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
  set_target_properties(
    a.out
    PROPERTIES BUILD_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs
  )
  set_target_properties(
    a.out
    PROPERTIES INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs
  )
endif()
if(WIN32 AND BASIX_PY_DIR)
  # Windows has no rpath: copy basix.dll and the delvewheel-vendored
  # dependencies next to the executable instead.
  file(
    GLOB _basix_dlls
    ${BASIX_PY_DIR}/*.dll
    ${BASIX_PY_DIR}/../fenics_basix.libs/*.dll
  )
  add_custom_command(
    TARGET a.out
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${_basix_dlls}
      $<TARGET_FILE_DIR:a.out>
  )
endif()

target_link_libraries(a.out PRIVATE Basix::basix)
