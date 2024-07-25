# X-Ray Toolchain 

The X-Ray static code anaylsis toolchain consists of three project repositories:

* code-parser
* code-detector
* coderrect

A brief program workflow:

<img src="./docs/images/workflow.jpg" width="1100px">

<br/>

## Supported Solana Bug Pattern List

The supported bug pattern list for Solana contract is saved in [coderrect.json](./coderrect/package/conf/coderrect.json). The format is as below:

```
"10001": {
      "name": "ReentrancyEtherVulnerability",
      "description": "The function may suffer from reentrancy attacks due to the use of call.value ...",
      "url": "https://consensys.github.io/smart-contract-best-practices/attacks/reentrancy/",
      "free": true
}
```
<br/>

## Quick Installation

We provide a quick installation with Docker, using pre-built LLVM image:

```
docker build -f Dockerfile.llvm -t llvm-prebuilt .
docker build -f Dockerfile.x-ray -t x-ray:latest .
docker run -it --name x-ray-1 x-ray:latest
```
If you want to build x-ray manually, please refer to [Manual Installation](./README.md#manual-installation) 

<br/>

## Start a scan

In order to scan a target repository, simply clone it and run `coderrect` it in the project's root directory (no need to look for the source code location)

As a demo, we choose [cashio](https://github.com/cashioapp/cashio), a stablecoin based Solana, as our target repo:

```
git clone https://github.com/cashioapp/cashio
cd cashio
coderrect -t .
```

To see a demo result, please visit: [demo result](./demo/README.md) (this demo uses [jet-v1](https://github.com/jet-lab/jet-v1) as test target)

<br/>

## Troubleshooting

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

<br/>

---

<br/>

## Manual Installation

Here's a simplified instruction for manually build and run x-ray tool on your local environment:

1. Build [LLVM](https://github.com/llvm/llvm-project/tree/release/12.x) locally, and set $LLVM_DIR and $MLIR_DIR
2. Compile `code-parser` and `code-detector`, our analysis tool specified for Rust language
3. Compile `coderrect`, and set $PATH

### Dependencies

- LLVM 14.0.6 (support of higher version is WIP)
- Cmake 3.26 or higher
- Go 1.22 or higher
- ANTLR 4.13.1 (only needed for parser development)

### Update CMake

You may need to install CMake manually because CMake version from apt-get is lower than `code-detector` required.

If CMake is already installed with apt-get, remove it first:

```
apt remove cmake
```

Download CMake source code and build:

```
wget https://github.com/Kitware/CMake/releases/download/v3.26.0-rc1/cmake-3.26.0-rc1.tar.gz
tar xzvf cmake*.gz
cd cmake-3.26.0-rc1
./bootstrap
make
make install
```

If it notifies `missing OpenSSL`:

```
apt install libssl-dev
```

### Build a local LLVM

The current Coderrect toolchain is based on LLVM version of 12.x.

```
git clone https://github.com/llvm/llvm-project.git
mkdir -p llvm-project/build && cd llvm-project/build
git checkout release/12.x
cmake -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt;lld;mlir" ../llvm/ -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
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


## Addendum

WIP: suuport of LLVM 17 and later version. [doc](./docs/latest-LLVM-support.md). 



