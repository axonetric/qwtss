cmake_minimum_required(VERSION 3.28)

project(qwtss LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "all-major") # Builds a "fat binary" that contains compiled code for all major, modern NVIDIA architectures

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

# --- CORE LIBRARY ---------------------------------------------------------------------------------------------------------_
add_library(qwtss_core STATIC
    src/core/CryptoUtils.h
    src/core/FastPoseidonHashCpu.cpp
    src/core/LabbeJR11Oracle.cpp
    src/core/PoseidonConstants.h
    src/core/PrivateKeyDatabase.h
    src/core/QwtssAir.h
    src/core/QwtssAirTypes.cpp
    src/core/QwtssConfig.h
    src/core/QwtssCore.cu
    src/core/QwtssCoreCpu.cpp
    src/core/QwtssCoreGpuOptimized.cu
    src/core/QwtssCoreShared.cu
    src/core/StarkTrace.cpp
    src/core/Tiles.h
)

target_include_directories(qwtss_core PUBLIC src/core)

# Link OpenMP immediately to trigger the proper compile options next
target_link_libraries(qwtss_core PRIVATE OpenMP::OpenMP_CXX)

# Aggressive Compiler Optimizations (CPU & GPU)
target_compile_options(qwtss_core PRIVATE
    # CPU: Max optimization + native instruction sets
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    # GPU: Max optimization + Fast Math (flushes denormals to zero, uses hardware intrinsics)
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>

    # CPU: Generate debug symbols, disable optimization
    #$<$<COMPILE_LANGUAGE:CXX>:-g -O0 -march=native>
    # GPU: Generate device debug symbols
    #$<$<COMPILE_LANGUAGE:CUDA>:-g -G>
)

# Define NDEBUG so Stone Prover headers allow default constructors
target_compile_definitions(qwtss_core PUBLIC NDEBUG)

# --- STONE PROVER INTEGRATION ---

# 1. Define the root path to the Stone Prover clone
set(STONE_PROVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Stone-Prover/stone-prover")

# 2. Add the include directories so source files can find the StarkWare headers
target_include_directories(qwtss_core SYSTEM PUBLIC
    ${STONE_PROVER_DIR}/src
    ${STONE_PROVER_DIR}/bazel-stone-prover/external/jsoncpp/include
)

# 3. Recursively find all the compiled static libraries (.a files)
# Step A: Ask Bazel directly for the absolute, fully-resolved output path
execute_process(
    COMMAND bazel info bazel-bin
    WORKING_DIRECTORY ${STONE_PROVER_DIR}
    OUTPUT_VARIABLE BAZEL_BIN_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Resolved Bazel Bin Directory: ${BAZEL_BIN_DIR}")

# Step B: Search that exact absolute path for the libraries
execute_process(
    COMMAND find -L ${BAZEL_BIN_DIR} -type f -name "*.a"
    OUTPUT_VARIABLE STONE_LIBS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert the raw newline-separated output into a properly formatted CMake list
string(REPLACE "\n" ";" STONE_LIBS "${STONE_LIBS_RAW}")

# Remove empty elements (if any) to prevent CMake linking errors
list(REMOVE_ITEM STONE_LIBS "")

# Debugging Check: Print to the terminal
list(LENGTH STONE_LIBS NUM_STONE_LIBS)
message(STATUS "==========> FOUND ${NUM_STONE_LIBS} STONE PROVER LIBRARIES <==========")

# Link the STARK libraries and the required GMP system library
target_link_libraries(qwtss_core PRIVATE 
    # Force the linker to circularly resolve dependencies among the 50+ Bazel static libs
    -Wl,--start-group
    ${STONE_LIBS} 
    -Wl,--end-group

    gmp
    gmpxx               # C++ GMP wrapper (for FastPoseidonHash.cpp)
    glog                # Google Logging (Required by Stone Prover)
    gflags              # Google Flags (Required by Stone Prover)
    dw                  # Prerequisite: DWARF library for Stone Prover's stack traces
    CUDA::cudart        # CUDA Runtime
    pthread             # Required by TaskManager
    #OpenMP::OpenMP_CXX # Added earlier above in separate call
    Boost::boost
)

# --- CLI EXECUTABLE (Public Release) ---------------------------------------------------------------------------------------
add_executable(qwtss-cli src/cli/QwtssCLI.cpp)
target_link_libraries(qwtss-cli PRIVATE qwtss_core)

# --- TESTS EXECUTABLE ------------------------------------------------------------------------------------------------------
add_executable(qwtss-tests src/tests/QwtssSystemTests.cpp)
target_link_libraries(qwtss-tests PRIVATE qwtss_core)

# --- RESEARCH EXECUTABLE ------------------------------------------------------------------------------------------------------
add_executable(qwtss-research
    src/research/main.cpp
    src/research/QwtssCryptanalysis.cu
    src/research/QwtssCryptanalysisCore.cu
    src/research/QwtssExtra.cpp
    src/research/QwtssPoW.cu
)
target_link_libraries(qwtss-research PRIVATE qwtss_core)
