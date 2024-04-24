#add_definitions(-DDebug)
set(Sources SmalltalkListener.cpp SmalltalkLexer.cpp SmalltalkParser.cpp)

add_library(stparserlib STATIC ${Sources})

target_link_libraries(stparserlib ${ANTLR4CPP_LOCAL_DIR}/lib/libantlr4-runtime.a)
#target_link_libraries(stparserlib "/usr/local/lib/libantlr4-runtime.so")