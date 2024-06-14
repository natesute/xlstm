cmake_minimum_required(VERSION 3.18)  # Recommended for better CUDA support
project(xLSTM LANGUAGES CXX CUDA)  # Explicitly state CUDA as a project language

set(CMAKE_CUDA_ARCHITECTURES 80)  # Ensure your hardware supports this

# Add subdirectories containing project modules
add_subdirectory(cuda)
add_subdirectory(cpp)

# Executable targets for tests
add_executable(test_slstm cpp/tests/test_slstm.cpp)
target_link_libraries(test_slstm xlstm_cpp)

add_executable(test_mlstm cpp/tests/test_mlstm.cpp)
target_link_libraries(test_mlstm xlstm_cpp)

add_executable(test_xlstm_block cpp/tests/test_xlstm_block.cpp)
target_link_libraries(test_xlstm_block xlstm_cpp)

add_executable(test_xlstm_model cpp/tests/test_xlstm_model.cpp)
target_link_libraries(test_xlstm_model xlstm_cpp)

# Executable targets for examples
add_executable(example_slstm cpp/examples/example_slstm.cpp)
target_link_libraries(example_slstm xlstm_cpp)

add_executable(example_mlstm cpp/examples/example_mlstm.cpp)
target_link_libraries(example_mlstm xlstm_cpp)

add_executable(example_xlstm_block cpp/examples/example_xlstm_block.cpp)
target_link_libraries(example_xlstm_block xlstm_cpp)

add_executable(example_xlstm_model cpp/examples/example_xlstm_model.cpp)
target_link_libraries(example_xlstm_model xlstm_cpp)

# Define include directories for the xlstm_cpp library
target_include_directories(xlstm_cpp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/layers
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/blocks
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/models
)

# Link CUDA libraries to xlstm_cpp (assuming xlstm_cuda is the CUDA library target)
target_link_libraries(xlstm_cpp xlstm_cuda)

# If xlstm_cuda uses any CUDA runtime libraries, link them here
find_package(CUDAToolkit REQUIRED)
target_link_libraries(xlstm_cpp CUDA::cudart)
