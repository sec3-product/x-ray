set(UnitTests
  Main.cpp
  RaceDetector/SVFTest.cpp
  RaceDetector/RDUtilTest.cpp)

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
  transformutils
  # needed when link llvm as shared library
  AggressiveInstCombine
  demangle
  mc)


add_executable(cr_unittest ${UnitTests})

include_directories(
  ${CMAKE_SOURCE_DIR}/tools/RaceDetector/include
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/utils/jsoncons/include
  ${CMAKE_SOURCE_DIR}/utils/googletest/include
)

ExternalProject_Get_Property(z3 INSTALL_DIR)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(BUILD_SHARED_Z3 TRUE) #for faster link in debug build
else()
  set(BUILD_SHARED_Z3 FALSE)
endif()

target_link_libraries(cr_unittest aser_gtest_main)
target_link_libraries(cr_unittest racedetect-lib)
target_link_libraries(cr_unittest aser)
target_link_libraries(cr_unittest ${llvm_libs})

if(${BUILD_SHARED_Z3})
  target_link_libraries(cr_unittest ${INSTALL_DIR}/lib/libz3.so)
else()
  target_link_libraries(cr_unittest ${INSTALL_DIR}/lib/libz3.a)
endif()

add_test(
  NAME cr_test
  COMMAND ${CMAKE_BINARY_DIR}/unittests/cr_unittest
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests
)

add_subdirectory(PointerAnalysis)
