set(SOURCE_FILES
    sol/MLIRGen.cpp
    sol/LowerToLLVM.cpp
)

add_library(SolRaceDetect STATIC ${SOURCE_FILES})

target_include_directories(SolRaceDetect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(SolRaceDetect
    conflib
)

# Enable exceptions for SolRaceDetect target, introduced via
# external/antlr/antlr4cpp.
target_compile_options(SolRaceDetect PRIVATE -fexceptions)

if (ENABLE_BUILD_ANTLR)
    set(ANTLR4CPP_JAR_LOCATION ${CMAKE_SOURCE_DIR}/external/antlr/antlr-4.13.1-complete.jar)
    LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
    # add external build for antlrcpp
    include(ExternalAntlr4Cpp)
    # Include the runtime to compile against
    include_directories(${ANTLR4CPP_INCLUDE_DIRS})
    link_directories(${ANTLR4CPP_LIBS})
    message(STATUS "Found antlr4cpp libs: ${ANTLR4CPP_LIBS} and includes: ${ANTLR4CPP_INCLUDE_DIRS} ")

    # Call macro to add lexer and grammar to your build dependencies.
    # NOTE: Here, we define "antlrcpp" as our project's namespace
    antlr4cpp_process_grammar(demo antlrcpp
        ${CMAKE_SOURCE_DIR}/src/antlr-rust/RustLexer.g4
        ${CMAKE_SOURCE_DIR}/src/antlr-rust/RustParser.g4)
    # include generated files in project environment
    include_directories(${antlr4cpp_include_dirs_antlrcpp})
else()
    message(STATUS "antlr4cpp built already")
    set(ANTLR4CPP_PARSER_DIR ${CMAKE_SOURCE_DIR}/src/antlr-rust)
    include_directories(${ANTLR4CPP_PARSER_DIR})
endif()


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

add_executable(SolRaceDetectExe
    SolRaceDetector.cpp
    ${antlr4cpp_src_files_antlrcpp}
)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(SolRaceDetectExe
    SolRaceDetect
    AntlrRust
    o2
    conflib
    ${LLVM_LIBS}
    ${DIALECT_LIBS}
    ${CONVERSION_LIBS}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
)

if (ENABLE_HEAP_PROFILE)
    message("Heap Profiling using tcmalloc enabled")
    target_link_libraries(SolRaceDetectExe tcmalloc)
endif()

if (ENABLE_CPU_PROFILE)
    message("CPU Profiling using profiler enabled")
    target_link_libraries(SolRaceDetectExe profiler)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(SolRaceDetectExe -static-libstdc++)
endif()

set_target_properties(SolRaceDetectExe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(SolRaceDetectExe PROPERTIES OUTPUT_NAME "sol-racedetect")
