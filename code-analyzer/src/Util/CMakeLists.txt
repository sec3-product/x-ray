set(SOURCES
    TypeMetaData.cpp
    Util.cpp
    Log.cpp
)


add_library(Util STATIC ${SOURCES})

target_link_libraries(Util
    PointerAnalysis
    pthread
)

target_include_directories(Util
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
