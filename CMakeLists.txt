cmake_minimum_required(VERSION 3.18)
project(xLSTM LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Definitions
add_compile_definitions(USE_CUDA)

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80)

# Add subdirectories containing project modules
add_subdirectory(cuda)
add_subdirectory(cpp)

# Executable targets
function(add_xlstm_test_executables)
    foreach(target_name IN LISTS ARGV)
        add_executable(${target_name} cpp/tests/${target_name}.cpp)
        target_link_libraries(${target_name} PRIVATE xlstm_cpp)
    endforeach()
endfunction()

# Executable targets
function(add_xlstm_example_executables)
    foreach(target_name IN LISTS ARGV)
        add_executable(${target_name} cpp/examples/${target_name}.cpp)
        target_link_libraries(${target_name} PRIVATE xlstm_cpp)
    endforeach()
endfunction()

add_xlstm_test_executables(
    test_slstm
    test_mlstm
    test_xlstm_block
    test_xlstm_model
)

add_xlstm_example_executables(
    example_slstm
    example_mlstm
    example_xlstm_block
    example_xlstm_model
)

# Assuming xlstm_cpp is defined in cpp/CMakeLists.txt
# Append these lines after the subdirectory is added
target_include_directories(xlstm_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/blocks
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/utils)
target_link_libraries(xlstm_cpp PUBLIC xlstm_cuda CUDA::cudart)
