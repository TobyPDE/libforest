cmake_minimum_required(VERSION 2.6)

################################################################################
### PROJECT SETTINGS
################################################################################

project(libforest)

set(LIBF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

################################################################################
### CMAKE OPTIONS
################################################################################

# Let's you switch assertions on/off
# This must be switched on for the unit tests
OPTION(ENABLE_ASSERT "Enables assertions throughout the library" ON)
IF(ENABLE_ASSERT)
    ADD_DEFINITIONS(-DLIBF_ENABLE_ASSERT)
ENDIF(ENABLE_ASSERT)

################################################################################
### THIRD PARTY LIBRARIES
################################################################################

find_package(GTest)
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

# Add the necessary g++ flags:
# -std=c++11: C++11
# -g: Debuggin with Valgrind
# -Wall: All warnings
# OpenMP
set(CMAKE_CXX_FLAGS "-g -Wall ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
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
                    src/util.cpp 
                    src/error_handling.cpp)

target_link_libraries(libforest 
                    ${Boost_LIBRARIES} 
                    ${OpenCV_LIBRARIE}
                    ${OpenCV_LIBS})

# Build the examples
# TODO: Make this a CMake option or remove it completely from the build process
add_subdirectory(examples)

################################################################################
### SET UP UNIT TESTS
################################################################################

if(GTEST_FOUND)
    enable_testing()

    # Build the test suite
    add_executable(tests
                        tests/data.cpp
                        tests/util.cpp)
    target_link_libraries(tests
                        libforest 
                        ${GTEST_BOTH_LIBRARIES})
endif(GTEST_FOUND)
