cmake_minimum_required(VERSION 3.16)

# Set the version
project(BasixFull VERSION "0.3.1.0" LANGUAGES CXX)

# set(CMAKE_MACOSX_RPATH TRUE)

add_subdirectory(cpp)
add_subdirectory(python)
