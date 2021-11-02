cmake_minimum_required(VERSION 3.16)
project(basix-demo)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Require C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Basix REQUIRED)

# Executable
add_executable(${PROJECT_NAME} main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} Basix::basix)
