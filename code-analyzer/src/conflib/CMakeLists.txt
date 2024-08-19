set(Sources conflib.cpp)

add_library(conflib STATIC ${Sources})

target_include_directories(conflib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
