enable_testing()

add_executable(unit_tests
    SolLLVMIRGeneratorTest.cpp
    test_main.cpp
)

target_link_libraries(unit_tests
    SolCodeParser
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
