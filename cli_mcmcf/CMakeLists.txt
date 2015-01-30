cmake_minimum_required(VERSION 2.6)
project(cli_mcmcf)

add_subdirectory(lib/lib_mcmcf)

set(CMAKE_CXX_FLAGS "-std=c++0x")

file(GLOB SOURCES "src/*.cpp")

include_directories(lib/lib_mcmcf/include)
message(STATUS ${MCMCF_INCLUDE_DIRS})
add_executable(cli_mcmcf ${SOURCES})
target_link_libraries( cli_mcmcf mcmcf )
