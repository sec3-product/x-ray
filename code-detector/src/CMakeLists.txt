set(SOURCES
    PointerAnalysis/CMDOptions.cpp
    PointerAnalysis/Context/Context.cpp
    PointerAnalysis/Program/Program.cpp
    PointerAnalysis/Model/DefaultHeapModel.cpp
    PointerAnalysis/Model/DefaultLangModel.cpp
    PointerAnalysis/Model/FieldSensitive/FSLayouts.cpp
    PointerAnalysis/Model/MemModels.cpp
    PointerAnalysis/Model/CppMemModel/RewriteModeledAPIPass.cpp
    PointerAnalysis/Model/CppMemModel/SpecialObject/Vector.cpp
    PointerAnalysis/Model/CppMemModel/SpecialObject/VTablePtr.cpp
    PointerAnalysis/Solver/PointsToSet.cpp

    PreProcessing/Passes/CanonicalizeGEPPass.cpp
    PreProcessing/Passes/InsertGlobalCtorCallPass.cpp
    PreProcessing/Passes/RemoveExceptionHandlerPass.cpp
    PreProcessing/Passes/UnrollThreadCreateLoopPass.cpp
    PreProcessing/Passes/LoweringMemCpyPass.cpp
    PreProcessing/Passes/WrapperFunIdentifyPass.cpp
    PreProcessing/PreProcPassManagerBuilder.cpp
    PreProcessing/IRPreProcessor.cpp
    PreProcessing/Passes/RemoveASMInstPass.cpp
    PreProcessing/Passes/StandardHeapAPIRewritePass.cpp
    PreProcessing/Passes/TransformCallInstBitCastPass.cpp

    Util/TypeMetaData.cpp
    Util/Util.cpp
    Util/Log.cpp
)


add_library(aser STATIC ${SOURCES})

target_link_libraries(aser pthread)

target_include_directories(aser
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/PointerAnalysis/include
        ${CMAKE_CURRENT_SOURCE_DIR}/PreProcessing/include
        ${CMAKE_CURRENT_SOURCE_DIR}/Util/include
)

add_subdirectory(SolanaAnalyzor)
