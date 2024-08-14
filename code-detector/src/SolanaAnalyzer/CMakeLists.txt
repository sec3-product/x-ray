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

set(SOLANA_ANALYZER_SOURCES
    CustomAPIRewriters/RustAPIRewriter.cpp
    LocksetManager.cpp
    LogColor.cpp
    PTAModels/GraphBLASModel.cpp
    Rules/CosplayAccounts.cpp
    Rules/Races.cpp
    Rules/UnsafeOperation.cpp
    Rules/UntrustfulAccount.cpp
    SourceInfo.cpp
    StaticThread.cpp
    SVE.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/utils/jsoncons/include
)

add_library(SolanaAnalyzer ${SOLANA_ANALYZER_SOURCES})

target_link_libraries(SolanaAnalyzer
    PointerAnalysis
    Util
)

add_executable(SolanaAnalyzerExe
    main.cpp
    SolanaAnalysisPass.cpp
)

target_link_libraries(SolanaAnalyzerExe
    SolanaAnalyzer
    conflib
    ${LLVM_LIBS}
)

if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(SolanaAnalyzerExe -static-libstdc++)
endif()

set_target_properties(SolanaAnalyzerExe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(SolanaAnalyzerExe PROPERTIES OUTPUT_NAME "racedetect")

# Regenerate all the stubs to ensure they are updated.
add_custom_command(TARGET SolanaAnalyzerExe
    PRE_BUILD
    # COMMAND sh ${CMAKE_SOURCE_DIR}/stubs/generate_stubs.sh
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/stubs/bc ${CMAKE_BINARY_DIR}/bin/stub)
