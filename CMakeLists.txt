cmake_minimum_required(VERSION 3.18)

project(basix VERSION "0.0.1")
# set(CMAKE_BUILD_TYPE Debug)

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

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get pybind11 CMake dir
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT ECHO_OUTPUT_VARIABLE
  ECHO_ERROR_VARIABLE)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_cmake_dir}")

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen ${EIGEN3_INCLUDE_DIRS}")
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

find_package(xtensor REQUIRED)
# target_include_directories(your_target PUBLIC ${xtensor_INCLUDE_DIRS})
# target_link_libraries(your_target PUBLIC xtensor)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -Wno-comment" HAVE_EXTRA_FLAGS)
if (HAVE_EXTRA_FLAGS)
  set(CMAKE_CXX_FLAGS "-Wall -Werror -Wno-comment ${CMAKE_CXX_FLAGS}")
endif()




# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(cpp)

pybind11_add_module(_basixcpp MODULE cpp/wrapper.cpp)

set_target_properties(_basixcpp PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(_basixcpp PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

target_include_directories(basix PRIVATE ${xtensor_INCLUDE_DIRS})
target_include_directories(_basixcpp PRIVATE ${xtensor_INCLUDE_DIRS})


if (APPLE)
  set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "@loader_path/lib")
else()
  set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
endif()

target_link_libraries(_basixcpp PRIVATE pybind11::module basix)

set_target_properties(basix PROPERTIES PUBLIC_HEADER "cpp/basix.h")

install(TARGETS _basixcpp DESTINATION .)
install(TARGETS basix LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
