# C++ unit tests for the Basix library. Built against an installed Basix,
# in the same way as the demos in demo/cpp.
#
#   cmake -B build -S cpp/test
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# On Windows (a multi-config generator) pass a configuration that matches
# the installed Basix, e.g. `cmake --build build --config Release` and
# `ctest --test-dir build -C Release`.
cmake_minimum_required(VERSION 3.21)

project(basix-tests LANGUAGES CXX)

include(CTest)

# Use Python for detecting Basix when installed using combined build
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/BasixPythonHints.cmake)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(
  unittests
  cell.cpp
  finite-element.cpp
  polyset.cpp
  quadrature.cpp
  sobolev-spaces.cpp
)
target_link_libraries(unittests PRIVATE Basix::basix Catch2::Catch2WithMain)
target_compile_features(unittests PRIVATE cxx_std_20)
set_target_properties(unittests PROPERTIES CXX_EXTENSIONS OFF)

if(BASIX_PY_DIR AND IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
  set_target_properties(
    unittests
    PROPERTIES
      BUILD_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs
      INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs
  )
endif()

if(WIN32)
  # Windows has no rpath, so dlls have to sit next to the executable (or
  # on PATH). Copy the dlls of the imported targets that unittests links,
  # i.e. basix.dll and any vcpkg dependencies. `cmake -E true` is a no-op
  # for the case where there are none, since `copy_if_different` with no
  # source files is an error.
  add_custom_command(
    TARGET unittests
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E
      $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:unittests>>,copy_if_different,true>
      $<TARGET_RUNTIME_DLLS:unittests> $<TARGET_FILE_DIR:unittests>
    COMMAND_EXPAND_LISTS
  )

  # Dlls vendored into a wheel by delvewheel are not CMake targets, so
  # they are not covered by TARGET_RUNTIME_DLLS above.
  if(BASIX_PY_DIR)
    file(GLOB _basix_dlls ${BASIX_PY_DIR}/../fenics_basix.libs/*.dll)
    if(_basix_dlls)
      add_custom_command(
        TARGET unittests
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy_if_different ${_basix_dlls}
          $<TARGET_FILE_DIR:unittests>
      )
    endif()
  endif()
endif()

# Register each Catch2 test case with CTest. Discovery is deferred to test
# time so that configuring and building does not require the test binary to
# be runnable (e.g. before the dlls above have been copied).
catch_discover_tests(unittests DISCOVERY_MODE PRE_TEST)
