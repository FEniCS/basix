# Sets BASIX_PY_DIR to an installed `basix` Python package's directory,
# for use as a find_package(Basix HINTS) search hint. Shared by
# demo/cpp/ and test/test_cmake/.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
    OUTPUT_VARIABLE BASIX_PY_DIR
    RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BASIX_PY_DIR)
    message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
  endif()
endif()
