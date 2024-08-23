include(FetchContent)

FetchContent_Declare(
    jsoncons
    GIT_REPOSITORY https://github.com/danielaparker/jsoncons.git
    GIT_TAG v0.177.0
)
set(JSONCONS_BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(jsoncons)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.9.1
)
# Skip building its test (which doesn't compile).
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

add_subdirectory(conflib)
add_subdirectory(PointerAnalysis)
add_subdirectory(PreProcessing)
add_subdirectory(SolanaAnalyzer)
add_subdirectory(Util)
