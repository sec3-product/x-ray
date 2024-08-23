include(FetchContent)

FetchContent_Declare(
  jsoncons
  GIT_REPOSITORY https://github.com/danielaparker/jsoncons.git
  GIT_TAG v0.177.0
)
set(JSONCONS_BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(jsoncons)

add_subdirectory(antlr-rust)
add_subdirectory(conflib)
add_subdirectory(o2)
add_subdirectory(SolanaParser)
