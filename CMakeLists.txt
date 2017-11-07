cmake_minimum_required(VERSION 3.7)
project(BiAffine_2nd)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-std=c++11 -lboost_program_options -lboost_date_time -lboost_serialization -ldynet")
#set(SOURCE_FILES main.cpp utils.cpp lib/utils.h lib/k_means.cpp network.h config.h headers.h data.cpp lib/tarjan.cpp)
set(SOURCE_FILES main.cpp config.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/Users/tomoki/ClionProjects/BiAffine_2nd")


# look for Boost
if(DEFINED BOOST_ROOT OR DEFINED BOOSTROOT OR DEFINED ENV{BOOST_ROOT} OR DEFINED ENV{BOOSTROOT})
    set(Boost_NO_SYSTEM_PATHS ON)
    if(DEFINED ${Boost_INCLUDE_DIR})
        get_filename_component(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
    endif()
endif()
set(Boost_REALPATH ON)
set(Boost_USE_STATIC_LIB ON)
find_package(Boost COMPONENTS program_options serialization date_time REQUIRED)
message("-- Boost dir is " ${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# look for Eigen
set(EIGEN3_INCLUDE_DIR /Users/tomoki/eigen)
get_filename_component(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Eigen dir is " ${EIGEN3_INCLUDE_DIR})
#find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# look for Dynet
set(DYNET_INCLUDE_DIR /Users/tomoki/Desktop/dynet)
get_filename_component(DYNET_INCLUDE_DIR "${DYNET_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Dynet dir is " ${DYNET_INCLUDE_DIR})
include_directories(${DYNET_INCLUDE_DIR})
link_directories(${DYNET_INCLUDE_DIR}/build/dynet)

# look for util
set(UTIL_INCLUDE_DIR /Users/tomoki/Desktop/dynet/examples/cpp/utils)
get_filename_component(UTIL_INCLUDE_DIR "${UTIL_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Util dir is " ${UTIL_INCLUDE_DIR})
include_directories(${UTIL_INCLUDE_DIR})
link_directories(${UTIL_INCLUDE_DIR})

#add_executable(BiAffine ${SOURCE_FILES})
add_executable(BiAffine_2nd ${SOURCE_FILES})
target_link_libraries(BiAffine_2nd ${Boost_LIBRARIES} dynet)
