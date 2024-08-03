llvm_map_components_to_libnames(llvm_libs
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

set(sol-racedetect-component
    src/st/LowerToLLVM.cpp
)

add_llvm_library(sol-racedetect-lib
    ${sol-racedetect-component}
)

target_include_directories(sol-racedetect-lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sol-racedetect-lib
    conflib
)

if (ENABLE_BUILD_ANTLR)
    set(ANTLR4CPP_JAR_LOCATION ${CMAKE_SOURCE_DIR}/external/antlr/antlr-4.13.1-complete.jar)
    LIST( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
    # add external build for antlrcpp
    include( ExternalAntlr4Cpp )
    # Include the runtime to compile against
    include_directories( ${ANTLR4CPP_INCLUDE_DIRS} )
    link_directories( ${ANTLR4CPP_LIBS} )
    message(STATUS "Found antlr4cpp libs: ${ANTLR4CPP_LIBS} and includes: ${ANTLR4CPP_INCLUDE_DIRS} ")

    # Call macro to add lexer and grammar to your build dependencies.
    # NOTE: Here, we define "antlrcpp" as our project's namespace
    antlr4cpp_process_grammar(demo antlrcpp
        ${CMAKE_SOURCE_DIR}/src/parser/SmalltalkLexer.g4
        ${CMAKE_SOURCE_DIR}/src/parser/SmalltalkParser.g4)
    #  ${CMAKE_CURRENT_SOURCE_DIR}/TLexer.g4
    #  ${CMAKE_CURRENT_SOURCE_DIR}/TParser.g4
    # include generated files in project environment
    include_directories(${antlr4cpp_include_dirs_antlrcpp})
else()
    message(STATUS "antlr4cpp built already")
    set(ANTLR4CPP_PARSER_DIR ${CMAKE_SOURCE_DIR}/src/parser)
    include_directories(${ANTLR4CPP_PARSER_DIR})
endif()


add_executable(sol-racedetect
    src/st/MLIRGen.cpp
    src/STRaceDetector.cpp
    ${antlr4cpp_src_files_antlrcpp}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(sol-racedetect
    sol-racedetect-lib
    o2
    conflib
    stparserlib
    ${llvm_libs}
    ${dialect_libs}
    ${conversion_libs}
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
    target_link_libraries(sol-racedetect tcmalloc)
endif()

if (ENABLE_CPU_PROFILE)
    message("CPU Profiling using profiler enabled")
    target_link_libraries(sol-racedetect profiler)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(sol-racedetect -static-libstdc++)
endif()

set_target_properties(sol-racedetect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
