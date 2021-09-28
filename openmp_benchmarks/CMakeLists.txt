cmake_minimum_required(VERSION 2.8.9)
project(OpenMPBench)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE OpenMPBench)
set(CMAKE_BUILD_TYPE Release)

option(BUILD_OFFLOADING "Build with offloading support" OFF)

include_directories(include)
include_directories(3rdparty/rapidjson/include)
file(GLOB SOURCES "src/*.cpp")

add_executable(${EXECUTABLE} ${SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXECUTABLE} PUBLIC OpenMP::OpenMP_CXX)
endif()

#build offloading benchmarks
if(BUILD_OFFLOADING)
    set(OFFLOAD_EXEC OpenMPOffloadBench)
    file(GLOB OFFLOAD_SOURCE "target/src/*.cpp")
    add_executable(${OFFLOAD_EXEC} ${OFFLOAD_SOURCE})

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_61")
        set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}  -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_61")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} -foffload=nvptx-none  -foffload=-lm -fno-fast-math -fno-associative-math")
        set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -foffload=nvptx-none  -foffload=-lm -fno-fast-math -fno-associative-math")
    endif()


endif()