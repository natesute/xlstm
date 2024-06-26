cmake_minimum_required(VERSION 3.8)
project(xlstm_cuda LANGUAGES CUDA CXX)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -lineinfo -Xcompiler -Wall,-Wextra")

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++14")

# Include directories
include_directories(utils)

# CUDA source files
set(CUDA_SOURCES
    kernels/slstm_kernels.cu
    kernels/mlstm_kernels.cu
    kernels/block_kernels.cu
)

# CUDA header files
set(CUDA_HEADERS
    kernels/slstm_kernels.h
    kernels/mlstm_kernels.h
    kernels/block_kernels.h
)

# CUDA library target
add_library(xlstm_cuda STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})

# Set CUDA separable compilation
set_target_properties(xlstm_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Specify include directories for the library target
target_include_directories(xlstm_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})