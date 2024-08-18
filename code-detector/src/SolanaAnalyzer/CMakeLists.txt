# Analyzer library.
set(SOLANA_ANALYZER_SOURCES
    AccountIDL.cpp
    Collectors/CosplayAccounts.cpp
    Collectors/Races.cpp
    Collectors/UnsafeOperation.cpp
    Collectors/UntrustfulAccount.cpp
    CustomAPIRewriters/RustAPIRewriter.cpp
    DebugFlags.cpp
    LocksetManager.cpp
    LogColor.cpp
    PTAModels/GraphBLASModel.cpp
    Rules/OverflowAdd.cpp
    Rules/Rule.cpp
    SolanaAnalysisPass.cpp
    SourceInfo.cpp
    StaticThread.cpp
    SVE.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/utils/jsoncons/include
)

add_library(SolanaAnalyzer ${SOLANA_ANALYZER_SOURCES})

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

target_link_libraries(SolanaAnalyzer
    PointerAnalysis
    Util
    ${LLVM_LIBS}
)

# Executable.
add_executable(SolanaAnalyzerExe
    Main.cpp
)

target_link_libraries(SolanaAnalyzerExe
    SolanaAnalyzer
    conflib
    Util
)

if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(SolanaAnalyzerExe -static-libstdc++)
endif()

set_target_properties(SolanaAnalyzerExe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(SolanaAnalyzerExe PROPERTIES OUTPUT_NAME "racedetect")
