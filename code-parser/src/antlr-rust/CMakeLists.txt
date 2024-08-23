set(SOURCES
    RustLexer.cpp
    RustParser.cpp
    RustParserBaseListener.cpp
    RustParserBaseVisitor.cpp
    RustParserListener.cpp
    RustParserVisitor.cpp
)

add_library(AntlrRust STATIC ${SOURCES})

target_link_libraries(AntlrRust
    ${ANTLR4CPP_LOCAL_DIR}/lib/libantlr4-runtime.a
)
