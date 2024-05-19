#add_definitions(-DDebug)
set(Sources
        Util/Log.cpp
        Util/Util.cpp
        )


add_library(o2 STATIC ${Sources})

target_link_libraries(o2 pthread)