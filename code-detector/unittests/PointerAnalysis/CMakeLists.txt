set(PTAUnitTest
        TestDriver.cpp
        PTAPassTest.cpp)

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
        demangle
        transformutils)

add_executable(pta_unittest ${PTAUnitTest})

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils/jsoncons/include
        ${CMAKE_SOURCE_DIR}/utils/googletest/include
)

target_link_libraries(pta_unittest aser_gtest_main) # google test
target_link_libraries(pta_unittest ${llvm_libs}) # llvm
target_link_libraries(pta_unittest aser) # pointer analysis

set(BIN_DIR ${CMAKE_BINARY_DIR}/unittests/PointerAnalysis)
set(TEST_CASES_DIR ${CMAKE_SOURCE_DIR}/unittests/PointerAnalysis/PTATestCases)

file(GLOB_RECURSE TEST_FILES "PTATestCases/*.ll")

foreach(FILE IN LISTS TEST_FILES)
    get_filename_component(FILE_NAME ${FILE} NAME_WE)
    message("adding pta test on ${FILE_NAME}")
    add_test(
            NAME pta_test_${FILE_NAME}
            COMMAND ${BIN_DIR}/pta_unittest ${FILE}
            WORKING_DIRECTORY ${BIN_DIR}
    )
endforeach()
#
#add_test(
#        NAME pta_test
#
#
#)
