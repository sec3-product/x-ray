enable_testing()

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests
    Main.cpp
)

target_link_libraries(unit_tests
    SolanaAnalyzer
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
