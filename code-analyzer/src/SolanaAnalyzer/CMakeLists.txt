# Analyzer library.
set(SOLANA_ANALYZER_SOURCES
    AccountIDL.cpp
    Collectors/CosplayAccounts.cpp
    Collectors/Output.cpp
    Collectors/UnsafeOperation.cpp
    Collectors/UntrustfulAccount.cpp
    DebugFlags.cpp
    LocksetManager.cpp
    LogColor.cpp
    PTAModels/GraphBLASModel.cpp
    Rules/OverflowAdd.cpp
    Rules/OverflowDiv.cpp
    Rules/OverflowMul.cpp
    Rules/OverflowSub.cpp
    Rules/Rule.cpp
    Rules/Ruleset.cpp
    RustAPIRewriter.cpp
    SolanaAnalysisPass.cpp
    SourceInfo.cpp
    StaticThread.cpp
    SVE.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(SolanaAnalyzer ${SOLANA_ANALYZER_SOURCES})

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.9.1
)
# Skip building its test (which doesn't compile).
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

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
    jsoncons
    nlohmann_json

    ${LLVM_LIBS}
)

target_include_directories(SolanaAnalyzer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Executable.
add_executable(SolanaAnalyzerExe
    Main.cpp
)

target_link_libraries(SolanaAnalyzerExe
    SolanaAnalyzer
    PointerAnalysis
    conflib
    Util
)

if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(SolanaAnalyzerExe -static-libstdc++)
endif()

set_target_properties(SolanaAnalyzerExe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(SolanaAnalyzerExe PROPERTIES OUTPUT_NAME "sol-code-analyzer")
