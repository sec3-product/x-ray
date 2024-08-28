set(SOURCES
    RustLexer.cpp
    RustParser.cpp
    RustParserBaseListener.cpp
    RustParserBaseVisitor.cpp
    RustParserListener.cpp
    RustParserVisitor.cpp
)

add_library(AntlrRust STATIC ${SOURCES})

message(STATUS "Using prebuilt antlr4cpp lib at ${ANTLR4CPP_LOCAL_DIR}")
target_link_libraries(AntlrRust
    ${ANTLR4CPP_LOCAL_DIR}/lib/libantlr4-runtime.a
)

target_include_directories(AntlrRust
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
