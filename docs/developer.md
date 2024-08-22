# Developer Guide

## Major Components and Workflow

### Components and Repo Layout

The X-Ray static code analysis toolchain consists of three major components:

*  *code-parser*: This component is responsible for parsing the target code and
   converting it into LLVM IR (Intermediate Representation) code. The input is
   typically Solana smart contracts. The code parser identifies relevant code
   sections, parses them via [ANTLR](https://www.antlr.org/) into MLIR, and
   generate LLVM IR.

*  *code-analyzer*: This component traverses the generated LLVM IR to locate
   suspect patterns and collect potential vulnerabilities. It applies a set of
   static analysis rules to the LLVM IR to identify security issues or other
   types of vulnerabilities within the code.

*  *CLI*: The command-line interface (CLI) handles user interactions. It
   processes user inputs, triggers the code parser and code detector, and
   generates reports. The output is available in JSON and text formats, which
   can be used for further inspection or integrated into other tools.

### Workflow

<img src="./images/workflow.jpg" width="1100px">

#### Rule Engine

TBA

#### Built-in Rules

TBA

## Building from Source

X-Ray supports two building and deployment options. You can choose either one
to analyze codebases.

-  Container Images: Building Docker images for cross-platform compatibility
   and easy distribution.
-  Native Binaries: Compiling the source code directly on your local machine
   for optimal performance.

Both options require LLVM binaries compiled from source code. The general
workflow is as follows:

1. Prepare LLVM artifacts: You can either build LLVM from source manually or
   use our provided prebuilt Docker images.
2. Build X-Ray Components (Analyzer, Detector and CLI): Use the compiled LLVM
   artifacts to build the X-Ray analyzer and detector components.

Below are the detailed steps for each option.

### Option 1: Build Container Images

You can refer to the `build-container-image` make target in
[Makefile](../Makefile).

#### Build or Use Prebuilt LLVM Image

We provide a prebuilt LLVM Docker image for convenience. You can use this image
to skip the LLVM building process.

```sh
docker pull registry.digitalocean.com/soteria/llvm-prebuilt-14.0.6:latest
```

TODO: Update the image URL once a public image is available.

Alternatively, you can run the following command to build LLVM from its source
code. The [`Dockerfile.llvm`](../Dockerfile.llvm) sets up the environment with
the necessary tools, fetches the LLVM source code, and runs the build
automatically.

```sh
make build-llvm-prebuilt-image
```

#### Build X-Ray Image

The following command builds the container image for X-Ray:

```sh
make build-container-image
```

Once having the built `x-ray` image, you can run it with the following commands
to scan a given target repo (e.g. `./demo`). For complete usage, please refer
to the [Usage](../README.md#usage) section in the main README file.

```sh
mkdir -p workspace
docker run --rm -v $(pwd)/demo:/workspace/demo /workspace/demo/jet-v1
```

### Option 2: Build Native Binaries

Follow these steps to build and run the X-Ray toolchain locally. Compared to
container builds, native builds are faster but require more setup and
dependency management. The following steps will guide you through the necessary
dependencies setup, obtaining LLVM, and how to build X-Ray components.

#### Prerequisites

- CMake 3.24 or higher
- Go 1.22 or higher
- Optional: Clang compiler 14.0.6 (only required when compiling the initial
  LLVM source code locally; subsequent X-Ray component compilation will use the
  Clang compiler from the prebuilt/built LLVM)

Since most of the above prequisites are standard tools, detailed installation
steps are provided in the [troubleshooting section](#Troubleshooting).

#### Prepare LLVM Binaries

Before building X-Ray, you need to prepare the LLVM build artifacts.

We provide prebuilt LLVM binaries for convenience. You can extract the LLVM
build artifacts from our provided Docker images. Run the following commands to
pull the Docker image and extract the LLVM build artifacts into the build/llvm
directory:

```sh
docker pull registry.digitalocean.com/soteria/llvm-prebuilt-14.0.6:latest
make extract-llvm
```

If you prefer to build LLVM from source, ensure that the LLVM build artifacts
are placed in the `build/llvm` directory.

#### Build and Install X-Ray Components

Use the following commands to build the X-Ray components: code parser, code
analyzer and CLI.

```sh
make clean && make build-x-ray
```

After completing the build process, run the following command to install the
X-Ray artifacts. By default, the artifacts are installed in the `build/dist/`
directory. This can be overridden by setting the `INSTALL_DIR` environment
variable.

```sh
make install
```

### Troubleshooting

#### Killed (program cc1plus)

When using Docker, if the compiler output shows `"internal compiler error:
Killed (program cc1plus)"`, some threads might have been killed by the
OOM-killer due to excessive use of Docker VM resources.

Solution 1: Adjust `-j $(nproc)` and try to reduce the number of threads.

Solution 2: Maximize the CPU core and RAM allocation in Docker's resource
Settings.

#### Update CMake

You may need to install CMake manually because the version available from the
default APT repository might be lower than required.

Follow the CMake download guide [here](https://cmake.org/download/) to install
the latest CMake version.

For Ubuntu, you can follow [this guide](https://apt.kitware.com/) to install
the latest CMake via APT. Below is a short version of the guide, which adds the
Kitware APT repository and installs the latest CMake version.

```sh
curl -sSf https://apt.kitware.com/kitware-archive.sh | sudo sh
sudo apt install cmake
```

#### Install Clang Compiler

Follow the official Clang installation guide [here](https://apt.llvm.org/). For
Ubuntu distros, the command looks as follows.

```sh
sudo apt install clang-14
```

#### Install Go

Follow the official Go installation guide [here](https://golang.org/doc/install).

## Code Style

We prefer to follow the LLVM coding style. A
[`.clang-format`](../.clang-format) style file is provided in the repository
root.

## Addendum

- [ ] Support for LLVM 17 and later versions is WIP. Refer to the
  [doc](./docs/latest-LLVM-support.md) for more details.
