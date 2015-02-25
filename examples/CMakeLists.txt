cmake_minimum_required(VERSION 2.6)
project(cli_libforest)

set(CMAKE_CXX_FLAGS "-std=c++0x")

find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(../include ${Boost_INCLUDE_DIRS})

add_executable(cli_mnist_rf mnist_rf.cpp)
target_link_libraries(cli_mnist_rf libforest ${Boost_LIBRARIES})

add_executable(cli_mnist_adaboost mnist_adaboost.cpp)
target_link_libraries(cli_mnist_adaboost libforest ${Boost_LIBRARIES})

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")