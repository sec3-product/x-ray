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

set(st-racedetect-component
        src/CustomAPIRewriters/ThreadAPIRewriter.cpp
        src/CustomAPIRewriters/LockUnlockRewriter.cpp
        src/CustomAPIRewriters/HeapAllocAPIRewriter.cpp
        src/CustomAPIRewriters/ThreadProfileRewriter.cpp
        src/st/LowerToLLVM.cpp
        src/st/LinkModules.cpp
        )

include_directories(
        ${INSTALL_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
#add_library(st-racedetect-lib ${st-racedetect-component})

add_llvm_library(st-racedetect-lib
${st-racedetect-component}
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

#set(CMAKE_CXX_STANDARD_LIBRARIES "-lxml2" )
include_directories("/usr/include/libxml2/")
#link_libraries(libxml2.so)


if (ENABLE_BUILD_ANTLR)
    set(ANTLR4CPP_JAR_LOCATION ${CMAKE_SOURCE_DIR}/utils/antlr/antlr-4.9-complete.jar)
    LIST( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
    # add external build for antlrcpp
    include( ExternalAntlr4Cpp )
    # Include the runtime to compile against
    include_directories( ${ANTLR4CPP_INCLUDE_DIRS} )
    link_directories( ${ANTLR4CPP_LIBS} )
    message(STATUS "Found antlr4cpp libs: ${ANTLR4CPP_LIBS} and includes: ${ANTLR4CPP_INCLUDE_DIRS} ")

    # Call macro to add lexer and grammar to your build dependencies.
    # NOTE: Here, we define "antlrcpp" as our project's namespace
    antlr4cpp_process_grammar(demo antlrcpp
        ${CMAKE_SOURCE_DIR}/utils/parser/SmalltalkLexer.g4
        ${CMAKE_SOURCE_DIR}/utils/parser/SmalltalkParser.g4)
    #  ${CMAKE_CURRENT_SOURCE_DIR}/TLexer.g4
    #  ${CMAKE_CURRENT_SOURCE_DIR}/TParser.g4    
    # include generated files in project environment
    include_directories(${antlr4cpp_include_dirs_antlrcpp})
else()
    message(STATUS "antlr4cpp built already")
    set(ANTLR4CPP_PARSER_DIR ${CMAKE_SOURCE_DIR}/utils/parser/)
    include_directories(${ANTLR4CPP_PARSER_DIR})
endif()


add_executable(st-racedetect src/st/MLIRGen.cpp src/STRaceDetector.cpp ${antlr4cpp_src_files_antlrcpp})
#add_dependencies(st-racedetect antlr4cpp antlr4cpp_generation_antlrcpp)

# the st-racedetect needs to be linked before o2
target_link_libraries(st-racedetect st-racedetect-lib)
target_link_libraries(st-racedetect o2)
target_link_libraries(st-racedetect conflib)
target_link_libraries(st-racedetect stparserlib)
target_link_libraries(st-racedetect ${llvm_libs})
target_link_libraries(st-racedetect xml2)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(st-racedetect
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIR
    MLIRTransforms
    )

if (ENABLE_HEAP_PROFILE)
    message("Heap Profiling using tcmalloc enabled")
    target_link_libraries(st-racedetect tcmalloc)
endif()

if (ENABLE_CPU_PROFILE)
    message("CPU Profiling using profiler enabled")
    target_link_libraries(st-racedetect profiler)
endif()
  
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_link_libraries(st-racedetect -static-libstdc++)
endif()
  
set_target_properties(st-racedetect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)