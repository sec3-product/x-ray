include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators.git
    GIT_TAG v2.3
)
FetchContent_MakeAvailable(indicators)

set(SOURCES
    Logger.cpp
)

add_library(Logger STATIC ${SOURCES})

target_link_libraries(Logger
    indicators::indicators
    spdlog::spdlog
)

target_include_directories(Logger
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        SYSTEM $<BUILD_INTERFACE:${spdlog_SOURCE_DIR}/include>
)
