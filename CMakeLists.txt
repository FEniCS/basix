cmake_minimum_required(VERSION 3.19)

project(BasixFull VERSION "0.8.0" LANGUAGES CXX)
set(BASIX_FULL_SKBUILD TRUE)
add_subdirectory(cpp)
add_subdirectory(python)
