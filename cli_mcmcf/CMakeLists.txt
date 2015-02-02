cmake_minimum_required(VERSION 2.6)
project(test)

add_subdirectory(lib/libforest)

set(CMAKE_CXX_FLAGS "-std=c++0x")

file(GLOB SOURCES "src/*.cpp")

include_directories(lib/libforest/include)
message(STATUS ${LIBF_INCLUDE_DIRS})
add_executable(cli_mcmcf ${SOURCES})
target_link_libraries( cli_mcmcf libforest )
find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

