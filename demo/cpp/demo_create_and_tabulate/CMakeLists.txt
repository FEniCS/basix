cmake_minimum_required(VERSION 3.16)
project(demo_create_and_tabulate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} Basix::basix xtensor)
