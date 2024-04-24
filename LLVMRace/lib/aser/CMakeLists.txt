#add_definitions(-DDebug)
set(Sources
        PointerAnalysis/Context/Context.cpp
        PointerAnalysis/Model/MemModels.cpp
        PointerAnalysis/Solver/PointsToSet.cpp
        PointerAnalysis/Program/Program.cpp
        PointerAnalysis/Model/DefaultLangModel.cpp
        PointerAnalysis/Model/FieldSensitive/FSLayouts.cpp
        PointerAnalysis/Model/DefaultHeapModel.cpp
        PointerAnalysis/Model/CppMemModel/SpecialObject/Vector.cpp
        PointerAnalysis/Model/CppMemModel/SpecialObject/VTablePtr.cpp
        PointerAnalysis/CMDOptions.cpp
        Util/TypeMetaData.cpp
        Util/Util.cpp
        Util/Log.cpp

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
        PointerAnalysis/Model/CppMemModel/RewriteModeledAPIPass.cpp
        PreProcessing/Passes/TransformCallInstBitCastPass.cpp)


add_library(aser STATIC ${Sources})

target_link_libraries(aser pthread)