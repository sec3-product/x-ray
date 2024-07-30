# Building from Source

## Building Docker Images

### Building or Using Pre-Built LLVM Image

TBA

We provide a quick installation with Docker, using prebuilt LLVM image.

```sh
make build-llvm-prebuilt-image
```

### Building X-Ray Image

```sh
make build-image
```

Once having the built `x-ray` image, you can run it with the following commands
to scan a given target repo (e.g. `./demo`).

```sh
mkdir -p workspace
docker run --rm -v $(pwd)/demo:/workspace/demo /workspace/demo/jet-v1
```

## Buidling Binaries

Here is a simplified instruction for manually building and running the x-ray
tool in your local environment.

1. Build [LLVM](https://github.com/llvm/llvm-project/tree/llvmorg-14.0.6)
   locally.
2. Compile `code-parser` and `code-detector`, our analysis tools specifically for
   the Rust language.
3. Compile `coderrect` and update your `PATH` environment variable accordingly.

### Prerequisites

- CMake 3.27 or higher
- Go 1.22 or higher
- ANTLR 4.13.1 (only needed for parser development)
- Clang compiler 14.0.6

#### Update CMake

You may need to install CMake manually because the CMake version from APT
repository is likely lower than the required one.

Follow the CMake download guide [here](https://cmake.org/download/) to install
the latest CMake version.

For Ubuntu, you can follow [this guide](https://apt.kitware.com/) to install
the latest CMake via APT. A short version of the guide is as follows, which
adds the Kitware APT repository and installs the latest CMake version.

```sh
curl -sSf https://apt.kitware.com/kitware-archive.sh | sudo sh
sudo apt install cmake
```

#### Install Clang Compiler

TODO: Update the doc to include Clang installation instructions. And this
should be optional as developers can use the prebuilt LLVM.

#### Install Go

Follow the official Go installation guide [here](https://golang.org/doc/install).


### Build a local LLVM for X-Ray

The current Coderrect toolchain is based on LLVM version of 14.x.

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir -p llvm-project/build && cd llvm-project/build
git checkout tags/llvmorg-14.0.6
cmake -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt;lld;mlir" ../llvm/ -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

If CMAKE output shows `"Could NOT find ZLIB"`

```
apt install zlib1g-dev
```

If CMAKE output shows `"Could NOT find LibXml2"`

```
apt install libxml2-dev
```

**NOTE:** When using Docker, if complier output shows `"internal compiler error: Killed (program cc1plus)"`, the reason is some threads getting killed by the OOM-killer due to overuse of Docker VM resources. 

Solution 1: Remove `-j $(nproc)` and try single thread mode.

Solution 2: Maximize the CPU core & RAM size of Resources in Docker Settings.

<br/>


### Build code-detector

code-detector is a code scanning tool that scans the generated LLVM IR (Intermediate Representation) to get a vulnerability list.

To manually build code-detector, please see following steps:

#### 1. Set LLVM_DIR and MLIR_DIR path

Before running cmake, you need to modify `./code-detector/CMakeLists.txt` to specify the LLVM_DIR path. This depends on the `llvm-project` you previously built:

```
set(LLVM_DIR [your-path]/llvm-project/build/lib/cmake/llvm/)
set(MLIR_DIR [your-path]/llvm-project/build/lib/cmake/mlir/)
```

#### 2. Install

```
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Simply use `make` instead of `make -j xx` if you don't need multi-thread.

<br/>

#### Troubleshooting: z3 repo not found

Solution: Modify `./code-detector/tools/RaceDetector/CMakeLists.txt`, add this line:

```
GIT_REPOSITORY "https://github.com/Z3Prover/z3"
```

**NOTE**: the above line should be removed after first-time building, otherwise it will slowdown the compilation process.

<br/>


### Build code-parser

Code-parser is a source code parsing tool that converts target code into IR code.

To manually build code-parser, please see following steps:

#### 1. Set LLVM_DIR and MLIR_DIR path

Before running cmake, you need to modify `./code-parser/CMakeLists.txt` to specify the LLVM_DIR path. This depends on the `llvm-project` you previously built:

```
set(LLVM_DIR [your-path]/llvm-project/build/lib/cmake/llvm/)
set(MLIR_DIR [your-path]/llvm-project/build/lib/cmake/mlir/)
```

#### 2. Install

```
sudo apt-get install libxml2-dev
mkdir -p build
cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Simply use `make` instead of `make -j xx` if you don't need multi-thread.

<br/>

#### Troubleshooting: Error 2

Makefile Error 2 means "Unknown error."

```
[ 23%] Linking CXX static library libconflib.a
[ 23%] Built target conflib
make: *** [Makefile:130: all] Error 2
```

Solution: specify `release` when running cmake.

```
cmake --build . --target clean
cmake -DCMAKE_BUILD_TYPE=Release ..
```

<br/>


### Build coderrect

Coderrect is an integrated tool that includes features such as code analysis, generating LLVM IR code, bug detection, generating reports, etc.

To manually build coderrect tool, please see following steps:


#### 1. Install

Navigate to the Go directory and directly run the existing `run` command:

```
cd gosrc
./run
```

**NOTE:** use `run` instead of `make all`, this script will copy the compiled binaries, `coderrect.json` and `data reporter` to the path at `whereis coderrect`.

Note that only `coderrect.json` and the relevant binaries of coderrect are copied here. Binaries from code-parser and code-detector are not copied. You still need to manually create the link.

#### 2. Create Coderrect bin path and add to system $PATH

The [update-bin.sh](./coderrect/gosrc/tools/update-bin.sh) script locates the path using the `which coderrect` command.

For first time running, if `coderrect` binary doesn't exist, you can specify a path, such as `~/local/sec3/bin/`, then generate a DUMMY coderrect binary:

```
mkdir [your-path]/sec3/bin/
touch [your-path]/sec3/bin/coderrect
chmod 755 [your-path]/sec3/bin/coderrect
```

Then run `which` or `whereis` to confirm the path. For example:

```
$ whereis coderrect
coderrect: /home/[your-username]/local/sec3/bin/coderrect
```

Then add that path to `$PATH` and save it to `.bashrc`.

```
export PATH="[your-path]/sec3/bin:$PATH"
```

Then run the script `run` again, the new build `coderrect` binary will overwrite the dummy one. 

#### 3. create binary links for code-parser and code-detector

Create two soft links for `code-parser` and `code-detector` binaries, which should be put at the same path as `coderrect`.

```
ln -s ./code-detector/build/bin/racedetect [your-path]/sec3/bin/racedetect
ln -s ./code-parser/build/bin/sol-racedetect [your-path]/sec3/bin/sol-racedetect
```

If the new built binaries are not linked yet, you need to create new links for them. 

<br/>



### Troubleshooting

#### If prompted for missing binaries

```
bash: [your-path]/coderrect/gosrc/bin/sol-racedetect: No such file or directory
bash: [your-path]/coderrect/gosrc/bin/racedetect: No such file or directory
```

Solution:

1. Make sure code-parser and code-detector have been correctly built

2. Add `coderrect` to system $PATH, and create link of `sol-racedetect` and `racedetect` at this path. See [create binary links](./README.md#3-create-binary-links-for-code-parser-and-code-detector)  

#### If prompted for missing JSON

```
"open coderrect/gosrc/conf/coderrect.json: no such file or directory"
```

Solution 1: Copy JSON file to `coderrect` local path:

```
$ find -name coderrect.json
./coderrect/package/conf/coderrect.json
$ cp -r ./coderrect/package/conf/ ./gosrc/
```

Solution 2: Add `coderrect` to system $PATH, and run the `run` script again, which will copy the compiled bin to the PATH that `whereis coderrect` is located.

#### If compiling coderrect prompts ig/fs error

Solution: Upgrade Go to version 1.16 or higher.


## Addendum

[ ]: support of LLVM 17 and later version. [doc](./docs/latest-LLVM-support.md). 

