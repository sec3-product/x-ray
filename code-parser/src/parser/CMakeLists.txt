set(SOURCES
    RustLexer.cpp
    RustParser.cpp
    RustParserBaseListener.cpp
    RustParserBaseVisitor.cpp
    RustParserListener.cpp
    RustParserVisitor.cpp
)

add_library(SolParser STATIC ${SOURCES})

target_link_libraries(SolParser
    ${ANTLR4CPP_LOCAL_DIR}/lib/libantlr4-runtime.a
)
