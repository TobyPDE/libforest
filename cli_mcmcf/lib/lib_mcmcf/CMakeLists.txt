cmake_minimum_required(VERSION 2.6)
project(mcmcf)

set(MCMCF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")

include_directories(${MCMCF_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++0x")

find_package( Boost 1.40.0 COMPONENTS system REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

file(GLOB MCMCF_SOURCES "src/*.cpp")
add_library(mcmcf STATIC ${MCMCF_SOURCES})
target_link_libraries( mcmcf ${Boost_LIBRARIES} )
