cmake_minimum_required(VERSION 3.24.0)
project(SolanaAnalyzer)

option(ENABLE_SPACE_OPT "Whether to apply aggressive memory optimzation (which will make constraint graph harder to understand)" ON)
option(ENABLE_HEAP_PROFILE "Turn on heap profiling feature by linking to tcmalloc" OFF)

set(LLVM_VERSION "14.0.6" CACHE STRING "The LLVM version")

# set to path in Docker
set(LLVM_DIR "/usr/local/llvm/lib/cmake/llvm" CACHE STRING "Path to LLVM")
set(MLIR_DIR "/usr/local/llvm/lib/cmake/mlir" CACHE STRING "Path to MLIR")

# Do not override manually set LLVM_DIR
if(NOT DEFINED LLVM_DIR)
    message(STATUS "Searching for LLVM_DIR in default directories")
    if (EXISTS /llvm-project/build/lib/cmake/llvm/)
        # We are probably in the dev container
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            # here link to shared llvm
            set(LLVM_DIR /llvm-project/build-shared/lib/cmake/llvm/)
        else()
            set(LLVM_DIR /llvm-project/build/lib/cmake/llvm/)
        endif()
    endif()
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRESOLVE_FUNPTR_IMMEDIATELY")

if (ENABLE_SPACE_OPT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NODE_ID_FOR_CONSTRAINTS -DNO_ADDR_OF_FOR_OFFSET")  # log level = debug
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstandalone-debug -DSPDLOG_ACTIVE_LEVEL=2 -DRACE_DETECT_DEBUG ")  # log level = debug
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_ACTIVE_LEVEL=6")  # log level = warn
endif()

set(COMMON_CXX_FLAG "-fno-rtti -DSPDLOG_NO_EXCEPTIONS")

# To support both in- and out-of-source builds,
# we check for the presence of the add_llvm_loadable_module command.
# - if this command is not present, we are building out-of-source
if(NOT COMMAND add_llvm_loadable_module)
    if(DEFINED LLVM_DIR)
        # We need to match the build environment for LLVM:
        # In particular, we need C++11 and the -fno-rtti flag
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 ${COMMON_CXX_FLAG}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 ${COMMON_CXX_FLAG}")
        endif()

        find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG NO_DEFAULT_PATH)
        message("Fount LLVM in ${LLVM_CMAKE_DIR}")

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

include_directories(
    external
    external/jsoncons/include
)

add_subdirectory(src)
add_subdirectory(test)
