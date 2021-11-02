cmake_minimum_required(VERSION 3.16)
project(basix-demo)

find_package(Basix REQUIRED)

# Executable
add_executable(${PROJECT_NAME} main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} basix)
