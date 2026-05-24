cmake_minimum_required(VERSION 3.14...3.28)

project(bwsr_inference_core LANGUAGES CXX)

# Enforce C++17 standard strictly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Locate pybind11
find_package(pybind11 REQUIRED)

# Apply hardware-specific compilation optimizations
# -O3: Maximum optimization for speed
# -march=native: Enable hardware-specific instruction sets (e.g., AVX2, AVX-512)
# -ffast-math: Break strict IEEE 754 compliance for mathematical throughput
add_compile_options(-O3 -march=native -ffast-math)

# Define the Python extension module
pybind11_add_module(_core_models 
    src/bwsr_inference/core/bindings.cpp 
    src/bwsr_inference/core/models.cpp
)

# Specify the output directory to align with the Python package structure
set_target_properties(_core_models PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/bwsr_inference/core
)