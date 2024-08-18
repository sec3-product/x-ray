set(SOURCES
    Passes/CanonicalizeGEPPass.cpp
    Passes/LoweringMemCpyPass.cpp
)

add_library(PreProcessing STATIC ${SOURCES})

target_link_libraries(PreProcessing
    Util
)

target_include_directories(PreProcessing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
