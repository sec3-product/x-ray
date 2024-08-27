set(SOURCES
    TypeMetaData.cpp
    Util.cpp
)

add_library(Util STATIC ${SOURCES})

target_link_libraries(Util
    Logger
    PointerAnalysis
)

target_include_directories(Util
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
