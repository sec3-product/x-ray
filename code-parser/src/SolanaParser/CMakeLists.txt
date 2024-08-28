include(FetchContent)

FetchContent_Declare(
    toml11
    GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
    GIT_TAG v4.2.0
)

FetchContent_MakeAvailable(toml11)

llvm_map_components_to_libnames(LLVM_LIBS
    bitwriter
    core
    ipo
    irreader
    instcombine
    instrumentation
    target
    linker
    analysis
    scalaropts
    support
    transformutils
    codegen

    # needed when link llvm as shared library
    AggressiveInstCombine
    demangle
    mc
)

# Static library.
set(SOURCE_FILES
    sol/LowerToLLVM.cpp
    sol/MLIRGen.cpp
    sol/SolLLVMIRGenerator.cpp
)

add_library(SolCodeParser STATIC ${SOURCE_FILES})

target_compile_options(SolCodeParser
    PRIVATE
        -fexceptions                 # Required by external/antlr/antlr4cpp
        -Wno-covered-switch-default  # Silence the warnings from jsoncons.hpp
)

target_include_directories(SolCodeParser
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        SYSTEM $<BUILD_INTERFACE:${toml11_SOURCE_DIR}/include>
)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(SolCodeParser
    AntlrRust
    conflib
    Logger
    ${LLVM_LIBS}
    ${DIALECT_LIBS}
    ${CONVERSION_LIBS}
    GTest::gtest_main
)

# Executable.
add_executable(SolCodeParserExe
    Main.cpp
)

target_compile_options(SolCodeParserExe
    PRIVATE
        -Wno-covered-switch-default  # Silence warnings from jsoncons.hpp
)

target_link_libraries(SolCodeParserExe
    SolCodeParser
)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(SolCodeParserExe -static-libstdc++)
endif()

set_target_properties(SolCodeParserExe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(SolCodeParserExe PROPERTIES OUTPUT_NAME "sol-code-parser")
