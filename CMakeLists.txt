cmake_minimum_required(VERSION 2.6)

################################################################################
### PROJECT SETTINGS
################################################################################

project(libforest)

set(LIBF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

enable_testing()

################################################################################
### CMAKE OPTIONS
################################################################################

# Let's you switch assertions on/off
# This must be switched on for the unit tests
OPTION(ENABLE_ASSERT "Enables assertions throughout the library" OFF)
IF(ENABLE_ASSERT)
    ADD_DEFINITIONS(-DLIBF_ENABLE_ASSERT)
ENDIF(ENABLE_ASSERT)

################################################################################
### THIRD PARTY LIBRARIES
################################################################################

find_package(GTest REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(Eigen3 REQUIRED)
# TODO: For PixelImportanceTool, should not be required - should be optional
# via CMake option!
# TODO: Also for OpenCV matrix reader. Add general "OpenCV Package option"
find_package(OpenCV REQUIRED) 
# TODO: Add general MATLAB option for importing MATLAB mat files
find_package(OpenMP)

################################################################################
### SET UP BUILD ENVIRONMENT
################################################################################

# Add the include directories of the third party libraries
include_directories(
                    ${EIGEN3_INCLUDE_DIR} 
                    ${LIBF_INCLUDE_DIRS} 
                    ${Boost_INCLUDE_DIR} 
                    ${OpenCV_INCLUDE_DIRS} 
                    ${GTEST_INCLUDE_DIRS})

# Add the necessary gcc flags:
# -std=c++11: C++11
# -g: Debuggin with Valgrind
# -Wall: All warnings
# OpenMP
set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

################################################################################
### SET UP TARGETS
################################################################################

# Build the actual library
add_library(libforest STATIC 
                    src/data.cpp 
                    src/classifiers.cpp 
                    src/estimators.cpp
                    src/unsupervised_learning.cpp 
                    src/online_learning.cpp 
                    src/learning.cpp 
                    src/tools.cpp 
                    src/error_handling.cpp)

target_link_libraries(libforest 
                    ${Boost_LIBRARIES} 
                    ${OpenCV_LIBRARIES})

# Build the examples
# TODO: Make this a CMake option or remove it completely from the build process
add_subdirectory(examples)

message(STATUS "GTest_INCLUDE_DIR=${GTest_INCLUDE_DIR}")
message(STATUS "GTest_LIBRARIES=${GTest_LIBRARIES}")

# Build the test suite
add_executable(test_util
                    tests/util.cpp)
target_link_libraries(test_util
                    libforest 
                    ${GTEST_LIBRARIES})
