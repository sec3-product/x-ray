set(SOURCES
    CMDOptions.cpp
    Context/Context.cpp
    Model/CppMemModel/RewriteModeledAPIPass.cpp
    Model/CppMemModel/SpecialObject/Vector.cpp
    Model/CppMemModel/SpecialObject/VTablePtr.cpp
    Model/DefaultHeapModel.cpp
    Model/DefaultLangModel.cpp
    Model/FieldSensitive/FSLayouts.cpp
    Model/MemModels.cpp
    Program/Program.cpp
    Solver/PointsToSet.cpp
)


add_library(PointerAnalysis STATIC ${SOURCES})

target_link_libraries(PointerAnalysis
    PreProcessing
    Util
    pthread
)

target_include_directories(PointerAnalysis
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
