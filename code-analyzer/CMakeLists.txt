cmake_minimum_required(VERSION 3.24.0)
project(SolanaAnalyzer)

set(LLVM_VERSION "14.0.6" CACHE STRING "The LLVM version")

# The default path to LLVM and MLIR, to match the ones in Dockerfile.
set(LLVM_DIR "/usr/local/llvm/lib/cmake/llvm" CACHE STRING "Path to LLVM")
set(MLIR_DIR "/usr/local/llvm/lib/cmake/mlir" CACHE STRING "Path to MLIR")

# Do not override manually set LLVM_DIR.
if(NOT DEFINED LLVM_DIR)
    message(STATUS "Searching for LLVM_DIR in default directories")
    if (EXISTS /llvm-project/build/lib/cmake/llvm/)
        set(LLVM_DIR /llvm-project/build/lib/cmake/llvm/)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Used by PointerAnalysis/Solver/PartialUpdateSolver.h.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRESOLVE_FUNPTR_IMMEDIATELY")

# Used by PointerAnalysis/Graph/ConstraintGraph/CGNodeBase.h.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NODE_ID_FOR_CONSTRAINTS -DNO_ADDR_OF_FOR_OFFSET")

# Set spdlog logger flags. Use the overall minimum level as SPDLOG_ACTIVE_LEVEL
# so that debug logs are written to file logs.
set(SPDLOG_FLAGS "-DSPDLOG_NO_EXCEPTIONS -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG")

# To support both in- and out-of-source builds,
# we check for the presence of the add_llvm_loadable_module command.
# - if this command is not present, we are building out-of-source
if(NOT COMMAND add_llvm_loadable_module)
    if(DEFINED LLVM_DIR)
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fno-rtti ${SPDLOG_FLAGS}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -fno-rtti ${SPDLOG_FLAGS}")
        endif()

        find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG NO_DEFAULT_PATH)
        message("Found LLVM in ${LLVM_CMAKE_DIR}")

        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
        include(AddLLVM)

        add_definitions(${LLVM_DEFINITIONS})
        include_directories(${LLVM_INCLUDE_DIRS})

    else()
        message(FATAL_ERROR "\
WARNING: The LLVM_DIR var was not set (required for an out-of-source build)!\n\
Please set this to environment variable to point to the LLVM build directory\
(e.g. on linux: export LLVM_DIR=/path/to/llvm/build/dir)")
    endif()
else()
    set(IN_SOURCE_BUILD 1)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../common ${CMAKE_BINARY_DIR}/common-build)
add_subdirectory(src)
add_subdirectory(test)
