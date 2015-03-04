cmake_minimum_required(VERSION 2.6)
project(libforest)

set(CMAKE_CXX_FLAGS "-std=c++0x -g")

set(LIBF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
find_package(Boost COMPONENTS system REQUIRED)
# TODO: For PixelImportanceTool, should not be required - shoudl be optional
# via CMake option!
find_package(OpenCV REQUIRED) 
include_directories(${LIBF_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(libforest STATIC src/data.cpp src/classifiers.cpp src/online_learning.cpp src/learning.cpp src/tools.cpp)
target_link_libraries(libforest ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_subdirectory(examples)