cmake_minimum_required(VERSION 3.21)

project(basix_nanobind LANGUAGES CXX)

# See https://gitlab.kitware.com/cmake/cmake/-/issues/16414
if(TARGET basix)
  add_library(Basix::basix ALIAS basix)
else()
  find_package(Basix REQUIRED)
  if(CMAKE_BUILD_TYPE STREQUAL "Developer")
    # This enables warnings from Basix headers even though they are a SYSTEM
    # dependency.
    set_target_properties(Basix::basix PROPERTIES SYSTEM OFF)
  endif()
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

# Create the binding library
if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_SABI "STABLE_ABI")
endif()
nanobind_add_module(
  _basixcpp
  NB_SUPPRESS_WARNINGS
  ${NANOBIND_SABI}
  basix/wrappers/basix_wrappers.cpp
)
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

# Generate nanobind type stubs (.pyi). On Windows the module can only be
# imported once its dependent DLLs sit beside it, so stubs are generated at
# install time. The DLLs are installed for that step only and removed after,
# leaving delvewheel to vendor them.
if(WIN32)
  install(TARGETS _basixcpp LIBRARY DESTINATION basix)

  install(CODE "set(_basix_installed_before \${CMAKE_INSTALL_MANIFEST_FILES})")
  if(TARGET basix)
    set(_basix_dll_set dependencies)
  else()
    set(_basix_dll_set basix_stub_dependencies)
    install(
      IMPORTED_RUNTIME_ARTIFACTS Basix::basix
      RUNTIME_DEPENDENCY_SET ${_basix_dll_set}
      RUNTIME DESTINATION basix
    )
  endif()
  install(
    RUNTIME_DEPENDENCY_SET ${_basix_dll_set} DESTINATION basix
    PRE_EXCLUDE_REGEXES [[api-ms-win-.*]] [[ext-ms-.*]]
    POST_EXCLUDE_REGEXES [[.*(\\|/)system32(\\|/).*\.dll]]
  )
  nanobind_add_stub(
    _basixcpp_stub
    MODULE basix._basixcpp
    INSTALL_TIME
    VERBOSE
    OUTPUT
      basix/_basixcpp.pyi
  )

  install(
    CODE
      "
    set(_basix_stub_dlls \${CMAKE_INSTALL_MANIFEST_FILES})
    list(REMOVE_ITEM _basix_stub_dlls \${_basix_installed_before})
    list(FILTER _basix_stub_dlls INCLUDE REGEX \\.dll\$)
    file(REMOVE \${_basix_stub_dlls})
  "
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
