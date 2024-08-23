include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.5.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators.git
    GIT_TAG v2.3
)
FetchContent_MakeAvailable(indicators)

set(SOURCES
    TypeMetaData.cpp
    Util.cpp
    Log.cpp
)

add_library(Util STATIC ${SOURCES})

target_link_libraries(Util
    PointerAnalysis
    indicators::indicators
    spdlog::spdlog
)

target_include_directories(Util
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        SYSTEM $<BUILD_INTERFACE:${spdlog_SOURCE_DIR}/include>
)
