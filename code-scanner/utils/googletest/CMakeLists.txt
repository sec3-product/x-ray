set(GOOGLETEST_VERSION 1.10.0)

project(gtest VERSION ${GOOGLETEST_VERSION} LANGUAGES CXX C)

include(cmake/internal_utils.cmake)

config_compiler_and_linker()

set (gtest_build_include_dirs
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR})

include_directories(${gtest_build_include_dirs})

cxx_library(aser_gtest "${cxx_strict}" src/gtest-all.cc)
cxx_library(aser_gtest_main "${cxx_strict}" src/gtest_main.cc)

target_link_libraries(aser_gtest_main PUBLIC aser_gtest)