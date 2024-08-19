include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.5.0
)

FetchContent_MakeAvailable(spdlog)

set(Sources
    Util/Log.cpp
    Util/Util.cpp
)

add_library(o2 STATIC ${Sources})

target_include_directories(o2
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        SYSTEM $<BUILD_INTERFACE:${spdlog_SOURCE_DIR}/include>
)
