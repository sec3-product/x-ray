set(SOURCES conflib.cpp)

add_library(conflib STATIC ${SOURCES})

target_include_directories(conflib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(conflib
    PUBLIC
        jsoncons
)

target_compile_options(conflib
    PRIVATE
        -Wno-covered-switch-default  # Silence warnings from jsoncons.hpp
)
