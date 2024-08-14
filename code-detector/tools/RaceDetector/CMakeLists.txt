# add_definitions(-DDebug)
include(ExternalProject)

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
        codegen

        # needed when link llvm as shared library
        AggressiveInstCombine
        demangle
        mc)

set(racedetect-component
    src/CustomAPIRewriters/RustAPIRewriter.cpp
    src/LocksetManager.cpp
    src/PTAModels/GraphBLASModel.cpp
    src/RDUtil.cpp
    src/Rules/Races.cpp
    src/Rules/UntrustfulAccount.cpp
    src/SourceInfo.cpp
    src/StaticThread.cpp
    src/SVE.cpp
)

include_directories(
        ${INSTALL_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils/jsoncons/include
)

add_library(racedetect-lib ${racedetect-component})
add_dependencies(racedetect-lib aser)

add_executable(racedetect src/RaceDetector.cpp src/RaceDetectionPass.cpp)

# the racedetect needs to be linked before aser

target_link_libraries(racedetect racedetect-lib)
target_link_libraries(racedetect aser)
target_link_libraries(racedetect conflib)
target_link_libraries(racedetect ${llvm_libs})
#target_link_libraries(racedetect omp)

if (ENABLE_HEAP_PROFILE)
    message("Heap Profiling using tcmalloc enabled")
    target_link_libraries(racedetect tcmalloc)
endif ()

if (ENABLE_CPU_PROFILE)
    message("CPU Profiling using profiler enabled")
    target_link_libraries(racedetect profiler)
endif ()


if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(racedetect -static-libstdc++)
endif ()


set_target_properties(racedetect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#regenerate all the stubs to ensure they are updated
add_custom_command(TARGET racedetect
        PRE_BUILD
        # COMMAND sh ${CMAKE_SOURCE_DIR}/stubs/generate_stubs.sh
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/stubs/bc ${CMAKE_BINARY_DIR}/bin/stub)
