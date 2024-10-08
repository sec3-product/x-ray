# ANTLR

This directory contains files generated by [ANTLR](https://www.antlr.org) for
the C++ target. ANTLR (Another Tool for Language Recognition) is a powerful
parser generator used to read, process, and translate structured text or binary
files. In this context, it is used to generate C++ code from grammar files
specific to the Rust language, supporting the X-Ray toolchain’s parsing
process.

An example command looks as follows, which generates the C++ code from the
given grammar files (`Rust*er.g4`). The used jar file is available at the
[ANTLR download page](https://www.antlr.org/download.html).

```sh
java -jar /path/to/antlr/antlr-4.9-complete.jar -visitor -Dlanguage=Cpp Rust*er.g4
```

The generated files are then compiled into a static library, which is used by
the X-Ray toolchain to parse Rust source code. Note that the final executable
needs to link against ANTLR runtime libraries, which are maintained in
`code-parser/external/antlr/antlr4cpp`.

For more detailed information on ANLTR and the C++ target, you can refer to the
[official ANTLR
documentation](https://github.com/antlr/antlr4/blob/master/doc/cpp-target.md#c).
