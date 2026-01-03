cmake_minimum_required(VERSION 3.16)
project(QueueSimulator VERSION 1.0.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Simulation library
add_library(simulation_lib STATIC
    src/simulation.cpp
)

# Main executable
add_executable(queue_sim
    src/main.cpp
)
target_link_libraries(queue_sim PRIVATE simulation_lib)

# Installation
install(TARGETS queue_sim DESTINATION bin)
install(FILES include/simulation.hpp DESTINATION include)

# Output info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
