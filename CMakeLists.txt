cmake_minimum_required(VERSION 3.22)
project(ETL_einsum LANGUAGES CXX CUDA)

option(ENABLE_CPU "Enable CPU backend" ON)
option(ENABLE_GPU "Enable GPU backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure CMAKE_BUILD_TYPE has a value for single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()


# Force -O2 for non-Debug builds
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 -DNDEBUG")

set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O2 -DNDEBUG")

# Enable separable compilation and position-independent code for CUDA
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# === Environment-based dependency resolution ===
set(ENVIRONMENT "my_computer" CACHE STRING "Target environment (e.g., my_computer, bear)")

add_subdirectory(src)

# Final executable
# add_executable(bench_gpu bench_gpu.cpp)
# target_link_libraries(bench_gpu PRIVATE benchmarking_gpu)
# 
# add_executable(bench_cpu bench_cpu.cpp)
# target_link_libraries(bench_cpu PRIVATE benchmarking_cpu)
# 
# add_executable(correct check_correctness.cpp)
# target_link_libraries(correct PRIVATE correctness)
# 
# add_executable(profile_cpu profile_cpu.cpp)
# target_link_libraries(profile_cpu PRIVATE core cpu_impl)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE core)

