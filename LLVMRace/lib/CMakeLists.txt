#add_subdirectory(CUDD)
#add_definitions(-DDebug)
add_subdirectory(aser)
## Due to a mutual dependencies, all the sub projects of the SVG are merged here
## Otherwise it is impossible to load the dependencies in opt
## NOTE: if the SVF should be linked into opt, we should probably use the individual sub projects here, rather than the combined project
#set(SOURCES
#    Util/AnalysisUtil.cpp
#    Util/CPPUtil.cpp
#    Util/DataFlowUtil.cpp
#    Util/PTACallGraph.cpp
#    Util/RaceAnnotator.cpp
#    Util/ThreadCallGraph.cpp
#    Util/BreakConstantExpr.cpp
#    Util/Conditions.cpp
#    Util/ExtAPI.cpp
#    Util/PathCondAllocator.cpp
#    Util/PTAStat.cpp
#    Util/ThreadAPI.cpp
#    Util/SVFModule.cpp
#    MemoryModel/CtxConsG.cpp
#    MemoryModel/ConsG.cpp
#    MemoryModel/LocationSet.cpp
#    MemoryModel/LocMemModel.cpp
#    MemoryModel/MemModel.cpp
#    MemoryModel/PAGBuilder.cpp
#    MemoryModel/PAG.cpp
#    MemoryModel/CHA.cpp
#    MemoryModel/PointerAnalysis.cpp
#    MemoryModel/OriginPAG.cpp
#    MemoryModel/CallSitePAG.cpp
#    MSSA/MemPartition.cpp
#    MSSA/MemRegion.cpp
#    MSSA/MemSSA.cpp
#    MSSA/SVFGBuilder.cpp
#    MSSA/SVFG.cpp
#    MSSA/SVFGOPT.cpp
#    MSSA/SVFGStat.cpp
#    SABER/DoubleFreeChecker.cpp
#    SABER/FileChecker.cpp
#    SABER/LeakChecker.cpp
#    SABER/ProgSlice.cpp
#    SABER/SaberAnnotator.cpp
#    SABER/SaberCheckerAPI.cpp
#    SABER/SaberSVFGBuilder.cpp
#    SABER/SrcSnkDDA.cpp
#    WPA/AndersenBase.cpp
#    WPA/AndersenLCD.cpp
#    WPA/AndersenStat.cpp
#    WPA/AndersenWave.cpp
#    WPA/AndersenWaveDiff.cpp
#    WPA/AndersenWaveDiffWithType.cpp
#    WPA/FlowSensitive.cpp
#    WPA/FlowSensitiveStat.cpp
#    WPA/TypeAnalysis.cpp
#    WPA/WPAPass.cpp
#
#        MTA/FSMPTA.cpp
#        MTA/LockAnalysis.cpp
#        MTA/MHP.cpp
#        MTA/MTA.cpp
#        MTA/MTAAnnotator.cpp
#        MTA/MTAResultValidator.cpp
#        MTA/MTAStat.cpp
#        MTA/PCG.cpp
#        MTA/TCT.cpp
#
#        RaceDectectorBase/RaceDetectorBase.cpp
#        RaceDectectorBase/InsensitiveLockSet.cpp
#        RaceDectectorBase/SHBGraph.cpp
#
#        RaceDetectorOrigin/RaceDetectorOrigin.cpp
#        RaceDetectorOrigin/StaticThread.cpp
#        RaceDetectorOrigin/SHBTrace.cpp)
#
#add_llvm_loadable_module(Svf ${SOURCES})
#add_llvm_Library(LLVMSvf ${SOURCES})
#
#link_directories( ${CMAKE_BINARY_DIR}/lib/Cudd )
#llvm_map_components_to_libnames(llvm_libs bitwriter core ipo irreader instcombine instrumentation target linker analysis scalaropts support )
#target_link_libraries(LLVMSvf ${llvm_libs})
#if ( CMAKE_SYSTEM_NAME MATCHES "Darwin")
#    target_link_libraries(Svf LLVMCudd ${llvm_libs})
#else()
#    target_link_libraries(Svf ${llvm_libs})
#endif()
#
#target_link_libraries(Svf aser)
#
#if(DEFINED IN_SOURCE_BUILD)
#    add_dependencies(Svf intrinsics_gen)
#    add_dependencies(LLVMSvf intrinsics_gen)
#endif()
