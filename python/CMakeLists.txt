cmake_minimum_required(VERSION 3.21)

project(basix_nanobind LANGUAGES CXX)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/16414
if(TARGET basix)
  add_library(Basix::basix ALIAS basix)
else()
  find_package(Basix REQUIRED)
endif()

find_package(
  Python
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT}
  REQUIRED
)

# Options
include(FeatureSummary)

option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(
  ENABLE_CLANG_TIDY
  ENABLE_CLANG_TIDY
  "Run clang-tidy while building"
)

option(
  BASIX_SET_INSTALL_RPATH
  "Set rpath of _basixcpp to location of installed C++ library"
  ON
)
add_feature_info(
  BASIX_SET_INSTALL_RPATH
  BASIX_SET_INSTALL_RPATH
  "Set rpath of _basixcpp to location of installed C++ library"
)

option(
  BASIX_NANOBIND_SPLIT
  "Build _basixcpp in nanobind split mode (requires nanobind >= 3.0)"
  OFF
)
add_feature_info(
  BASIX_NANOBIND_SPLIT
  BASIX_NANOBIND_SPLIT
  "Build _basixcpp in nanobind split mode (requires nanobind >= 3.0)"
)

feature_summary(WHAT ALL)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# nanobind < 3.0 ignores BACKEND_MODULE silently.
if(BASIX_NANOBIND_SPLIT AND nanobind_VERSION VERSION_LESS 3.0)
  message(FATAL_ERROR "BASIX_NANOBIND_SPLIT requires nanobind >= 3.0.")
endif()

# Create the binding library
if(BASIX_NANOBIND_SPLIT)
  set(NANOBIND_MODE BACKEND_MODULE nanobind_backend)
elseif(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_MODE STABLE_ABI)
endif()
nanobind_add_module(_basixcpp NB_SUPPRESS_WARNINGS ${NANOBIND_MODE} wrapper.cpp)
target_compile_features(_basixcpp PRIVATE cxx_std_20)

# Windows requires all symbols to be manually exported; this exports them
# automatically as on Unix. No-op on other platforms.
set_target_properties(_basixcpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    _basixcpp
    PROPERTIES
      CXX_CLANG_TIDY
        "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../.clang-tidy"
  )
endif()

target_link_libraries(_basixcpp PRIVATE Basix::basix)

# Add strict compiler flags
include(CheckCXXCompilerFlag)

set(BASIX_PY_CXX_DEVELOPER_FLAGS "")
set(BASIX_PY_CXX_DEVELOPER_DEFINITIONS "")

# Add some strict compiler checks
check_cxx_compiler_flag(
  "-Wall -Werror -Wextra -pedantic -Wno-comment"
  HAVE_PEDANTIC
)

if(HAVE_PEDANTIC)
  list(
    APPEND BASIX_PY_CXX_DEVELOPER_FLAGS
    -Wall;-Werror;-Wextra;-pedantic;-Wno-comment
  )
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)
if(HAVE_DEBUG)
  list(APPEND BASIX_PY_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)
if(HAVE_O2_OPTIMISATION)
  list(APPEND BASIX_PY_CXX_DEVELOPER_FLAGS -O2)
endif()

# Pipe
check_cxx_compiler_flag(-pipe HAVE_PIPE)
if(HAVE_PIPE)
  list(APPEND BASIX_PY_CXX_DEVELOPER_FLAGS -pipe)
endif()

# Enable C++ standard library debugging
include(CheckCXXSymbolExists)

check_cxx_symbol_exists(_LIBCPP_VERSION "version" LIBCPP)
check_cxx_symbol_exists(__GLIBCXX__ "version" GLIBCXX)

if(LIBCPP)
  list(
    APPEND BASIX_PY_CXX_DEVELOPER_DEFINITIONS
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
  )
endif()

if(GLIBCXX)
  list(APPEND BASIX_PY_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

target_compile_options(
  _basixcpp
  PRIVATE $<$<CONFIG:Developer>:${BASIX_PY_CXX_DEVELOPER_FLAGS}>
)
target_compile_definitions(
  _basixcpp
  PRIVATE $<$<CONFIG:Developer>:${BASIX_PY_CXX_DEVELOPER_DEFINITIONS}>
)

# TODO: remove after issue https://github.com/FEniCS/basix/issues/842 is resolved
target_compile_definitions(
  _basixcpp
  PRIVATE MDSPAN_USE_PAREN_OPERATOR=1 MDSPAN_USE_BRACKET_OPERATOR=0
)

if(BASIX_FULL_SKBUILD AND BASIX_SET_INSTALL_RPATH)
  # If building bundled C++ and Python wrapper wheel `pip install .` add relative
  # rpath so _basixcpp can find the bundled Basix::basix library when the wheel
  # is relocated.
  if(APPLE)
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "@loader_path/lib")
  elseif(UNIX)
    set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
  endif()
elseif(BASIX_SET_INSTALL_RPATH)
  # If building standalone Python wrapper `pip install python/` put absolute
  # path to Basix C++ library on rpath - can help linking if it is outside a
  # system directory. The build is non-relocatable.
  get_target_property(_location Basix::basix LOCATION)
  cmake_path(GET _location PARENT_PATH _basix_dir)
  set_target_properties(_basixcpp PROPERTIES INSTALL_RPATH ${_basix_dir})
endif()

# Generate nanobind type stubs (.pyi). On UNIX-like systems the freshly
# built module can be imported directly from the build tree (rpath
# resolves its dependencies), so stubs are generated at build time. On
# Windows, _basixcpp cannot be imported until it is installed next to
# its dependent DLLs, so stub generation is deferred to install time -
# nanobind installs the .pyi automatically in that mode.
if(WIN32)
  # This *must* be called prior to nanobind_add_stub
  install(TARGETS _basixcpp LIBRARY DESTINATION basix)

  # DEPENDS is redundant in INSTALL_TIME mode - _basixcpp must be installed.
  nanobind_add_stub(
      _basixcpp_stub
      MODULE basix._basixcpp
      INSTALL_TIME
      VERBOSE
      OUTPUT
        basix/_basixcpp.pyi
  )
else()
  nanobind_add_stub(
      _basixcpp_stub
      MODULE _basixcpp
      PYTHON_PATH $<TARGET_FILE_DIR:_basixcpp>
      DEPENDS _basixcpp
      VERBOSE
      OUTPUT
        _basixcpp.pyi
  )

  install(TARGETS _basixcpp LIBRARY DESTINATION basix)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_basixcpp.pyi DESTINATION basix)
endif()

# py.typed (PEP 561 marker) is not a tracked source file - write it
# directly at install time on both platforms rather than relying on
# nanobind's MARKER_FILE option, which requires importing the freshly
# built module (unreliable on Windows, see above) and would otherwise
# write a build artifact into the source tree on UNIX.
install(CODE "file(WRITE \"\${CMAKE_INSTALL_PREFIX}/basix/py.typed\" \"\")")
