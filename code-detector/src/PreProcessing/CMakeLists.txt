set(SOURCES
    IRPreProcessor.cpp
    Passes/CanonicalizeGEPPass.cpp
    Passes/InsertGlobalCtorCallPass.cpp
    Passes/LoweringMemCpyPass.cpp
    Passes/RemoveASMInstPass.cpp
    Passes/RemoveExceptionHandlerPass.cpp
    Passes/StandardHeapAPIRewritePass.cpp
    Passes/TransformCallInstBitCastPass.cpp
    Passes/UnrollThreadCreateLoopPass.cpp
    Passes/WrapperFunIdentifyPass.cpp
    PreProcPassManagerBuilder.cpp
)

add_library(PreProcessing STATIC ${SOURCES})

target_link_libraries(PreProcessing
    Util
    pthread
)

target_include_directories(PreProcessing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
