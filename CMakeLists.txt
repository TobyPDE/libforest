cmake_minimum_required(VERSION 2.6)
project(libforest)

set(LIBF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")

include_directories(${LIBF_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++0x")

find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

file(GLOB LIBF_SOURCES "src/*.cpp")
add_library(libforest STATIC ${LIBF_SOURCES})
target_link_libraries(libforest ${Boost_LIBRARIES})

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")