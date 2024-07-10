cmake_minimum_required(VERSION 3.4.3)
project(smalltalk)

option(ENABLE_HEAP_PROFILE "Turn on heap profiling feature by linking to tcmalloc" OFF)

# Do not override manually set LLVM_DIR
if(NOT DEFINED LLVM_DIR)
    message(STATUS "Searching for LLVM_DIR in default directories")
    if (EXISTS /llvm-project/build/lib/cmake/llvm/)
        set(LLVM_DIR /llvm-project/build/lib/cmake/llvm/)
    endif()
endif()

# use C++14 as llvm-11 does
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

set(LLVM_DIR /usr/local/llvm/lib/cmake/llvm)
set(MLIR_DIR /usr/local/llvm/lib/cmake/mlir)

if(NOT ENABLE_TEST)
    set(ENABLE_TEST Off) # default disable testing
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRESOLVE_FUNPTR_IMMEDIATELY")

if (ENABLE_SPACE_OPT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NODE_ID_FOR_CONSTRAINTS -DNO_ADDR_OF_FOR_OFFSET")  # log level = debug
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_ACTIVE_LEVEL=2 -DRACE_DETECT_DEBUG ")  # log level = debug
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_ACTIVE_LEVEL=6")  # log level = warn
endif()

#-fno-rtti 
set(COMMON_CXX_FLAG "-Wno-c++17-extensions -DSPDLOG_NO_EXCEPTIONS")

# To support both in- and out-of-source builds,
# we check for the presence of the add_llvm_loadable_module command.
# - if this command is not present, we are building out-of-source
if(NOT COMMAND add_llvm_loadable_module)
    if (DEFINED LLVM_DIR)
        # We need to match the build environment for LLVM:
        # In particular, we need C++11 and the -fno-rtti flag
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 ${COMMON_CXX_FLAG}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${COMMON_CXX_FLAG}")
        endif()

        find_package(LLVM 12.0.1 REQUIRED CONFIG NO_DEFAULT_PATH)
        # find_package(LLVM 17.0.0 REQUIRED CONFIG NO_DEFAULT_PATH)

        message("FIND LLVM IN ${LLVM_CMAKE_DIR}")

        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
        include(AddLLVM)
        include(TableGen)
        include(HandleLLVMOptions)

        add_definitions(${LLVM_DEFINITIONS})
        include_directories(${LLVM_INCLUDE_DIRS})

    else()
        message(FATAL_ERROR "\
WARNING: The LLVM_DIR var was not set (required for an out-of-source build)!\n\
Please set this to environment variable to point to the LLVM build directory\
(e.g. on linux: export LLVM_DIR=/path/to/llvm/build/dir)")
    endif()
    if (DEFINED MLIR_DIR)
    find_package(MLIR REQUIRED CONFIG NO_DEFAULT_PATH)
        message("FIND MLIR IN ${MLIR_CMAKE_DIR}")

        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        include(AddMLIR)
        add_definitions(${MLIR_DEFINITIONS})
        include_directories(${MLIR_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "\
WARNING: The MLIR_DIR var was not set (required for an out-of-source build)!\n\
Please set this to environment variable to point to the LLVM build directory\
(e.g. on linux: export MLIR_DIR=/path/to/mlir/build/dir)")
    endif()
else()
        set(IN_SOURCE_BUILD 1)
endif()

# set(ANTLR4CPP_LOCAL_DIR ${CMAKE_SOURCE_DIR}/utils/antlr/antlr4cpp)
#     list(APPEND ANTLR4CPP_INCLUDE_DIRS ${ANTLR4CPP_LOCAL_DIR}/include/antlr4-runtime)
#     foreach(src_path misc atn dfa tree support)
#         list(APPEND ANTLR4CPP_INCLUDE_DIRS ${ANTLR4CPP_LOCAL_DIR}/include/antlr4-runtime/${src_path})
#     endforeach(src_path)
#     include_directories( ${ANTLR4CPP_INCLUDE_DIRS} )
#     message(STATUS "Found antlr4cpp includes: ${ANTLR4CPP_INCLUDE_DIRS} ")
    
# ANTLR 4.13.1
set(ANTLR4CPP_LOCAL_DIR ${CMAKE_SOURCE_DIR}/utils/antlr/antlr4cpp)
    list(APPEND ANTLR4CPP_INCLUDE_DIRS ${ANTLR4CPP_LOCAL_DIR}/src)
    foreach(src_path misc atn dfa internal tree support)
        list(APPEND ANTLR4CPP_INCLUDE_DIRS ${ANTLR4CPP_LOCAL_DIR}/src/${src_path})
    endforeach(src_path)
    include_directories( ${ANTLR4CPP_INCLUDE_DIRS} )
    message(STATUS "Found antlr4cpp includes: ${ANTLR4CPP_INCLUDE_DIRS} ")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    utils/jsoncons/include
    utils)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(utils/conflib)
add_subdirectory(utils/parser)
