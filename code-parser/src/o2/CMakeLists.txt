set(Sources
    Util/Log.cpp
    Util/Util.cpp
)

add_library(o2 STATIC ${Sources})

target_include_directories(o2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(o2 pthread)
