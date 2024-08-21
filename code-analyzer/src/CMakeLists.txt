include(FetchContent)

FetchContent_Declare(
  jsoncons
  GIT_REPOSITORY https://github.com/danielaparker/jsoncons.git
  GIT_TAG v0.141.0
)

FetchContent_MakeAvailable(jsoncons)

add_subdirectory(conflib)
add_subdirectory(PointerAnalysis)
add_subdirectory(PreProcessing)
add_subdirectory(SolanaAnalyzer)
add_subdirectory(Util)
