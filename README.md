# X-Ray Toolchain 

The X-Ray static code anaylsis toolchain consists of three project repositories:

* [Coderrect](https://github.com/coderrect-inc/coderrect)
* [smallrace](https://github.com/coderrect-inc/smallrace)
* [LLVMRace](https://github.com/coderrect-inc/LLVMRace)

A brief program workflow:

<img src="./docs/images/workflow.jpg" width="1100px">

## Installation Steps

Here's a simplified intallation steps. The detailed installation and configuration is in [next chapter](.#installation).

1. Build [LLVM](https://github.com/llvm/llvm-project/tree/release/12.x) locally, and set $LLVM_DIR and $MLIR_DIR
2. Compile LLVMRace on the target language (Rust, Solidity, etc.)
3. Compile smallrace on the target language
4. Compile Coderrect, and set $PATH

**NOTE:** If the target repository's language changes, the three tools need to be switched to the corresponding target language branches (e.g., Rust, Solidity, etc.) and then be compiled again.

## Supported Bug Pattern List

The supported bug pattern list is saved in `coderrect.json` at corresponding branch, classfied by the target language. 

For example, the bug pattern for Solana contracts is:

https://github.com/coderrect-inc/coderrect/blob/rust/installer/package/conf/coderrect.json

<br/>

## Usage

In order to scan a target repository, simply clone it to local environment, and run `coderrect` it in the project's root directory (no need to look for the source code location)

As a demo, we choose [jet-v1](https://github.com/jet-lab/jet-v1), a smart contract on Solana, as our target repo:

```
git clone https://github.com/jet-lab/jet-v1
cd jet-v1
coderrect -t .
```
To see a demo result, please visit: [demo result](./demo/README.md)

<br/>

## Troubleshooting

#### If prompted for missing binaries
```
bash: /home/sec3/coderrect/gosrc/bin/sol-racedetect: No such file or directory
bash: /home/sec3/coderrect/gosrc/bin/racedetect: No such file or directory
```
Solution: 

1. Make sure LLVMRace and smallrace have been switched to the corresponding branch;

2. Add `coderrect` to system $PATH, and create link of `sol-racedetect` and `racedetect` at this path. See [create binary links](./README.md#4-create-binary-links-for-llvmrace-and-smallrace)  

#### If prompted for missing JSON
```
"open coderrect/gosrc/conf/coderrect.json: no such file or directory"
```
Solution 1: Copy JSON file to `coderrect` local path:
```
$ find -name coderrect.json
./installer/package/conf/coderrect.json
$ cp -r ./installer/package/conf/ ./gosrc/
```
Solution 2: Add `coderrect` to system $PATH, and run the `run` script again, which will copy the compiled bin to the PATH that `whereis coderrect` is located.

#### If compiling coderrect prompts ig/fs error

Solution: Upgrade Go to version 1.16 or higher.

<br/>

---

# Installation

### Dependencies

- LLVM 12.x (later version not tested)
- Cmake 3.26 or later
- Go 1.16 or later
- ANTLR 4.13.1 (only needed for parser development)

## Environment preparation

### Create a Docker container (optional)

You can use Docker to install Coderrect toolchain on a virtual environment.

**NOTE**: On a computer using Apple M series CPU (ARM architecture), if you pull a Docker image with x86 (a.k.a AMD64) architecture , a warning will appear: "amd64 images have poor performance, and sometimes crashing behavior as well". Due to some existing errors, Docker recommends avoiding use of amd64 images on Mac with Apple chips.

Pull an Ubuntu image, specifying ARM architecture:
```
docker pull ubuntu:20.04 --platform=linux/arm64
```
Run a Docker container with shared data folder:
```
docker run -i -t -v /Users/sec3/Git:/Git ubuntu:20.04 /bin/bash
```
Load a Docker container with a specified container ID (`61461e113c59` is an example here):

```
docker start 61461e113c59
docker exec -u -0 -it 61461e113c59 bash
```

### Update CMake

You may need to install CMake manually because CMake version from apt-get is lower than LLVM required.

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
mkdir -p llvm-project/build
cd llvm-project/build
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

---
<br/>

## LLVMRace

LLVMRace is a code scanning tool that scans the generated LLVM IR (Intermediate Representation) to get a vulnerability list.

### Switch Target Languages

LLVMRace also organizes target languages using git branches. Commonly used branches include:

- [rust](https://github.com/coderrect-inc/LLVMRace/tree/rust): support Rust language, for Solana projects
- [solidity](https://github.com/coderrect-inc/LLVMRace/tree/solidity): support Solidity language, for ETH projects
- [stx](https://github.com/coderrect-inc/LLVMRace/tree/stx): supports Smalltalk (a Domain Specific Language that our client company is using) projects

### Compilation and Configuration

Please see: [building-LLVMRace.md](./docs/building-LLVMRace/README.md)

<br/>

---
<br/>

## smallrace

Smallrace is a source code parsing tool that converts target code into IR code.

### Switching Target Languages

For the convenience of switching targe language, smallrace organizes and distinguishes target languages using git branches. 

Commonly used branches include:

- [rust](https://github.com/coderrect-inc/smallrace/tree/rust): support Rust language, for Solana projects
- [solidity](https://github.com/coderrect-inc/smallrace/tree/solidity): support Solidity language, for ETH projects
- [stx](https://github.com/coderrect-inc/smallrace/tree/stx): supports Smalltalk (a Domain Specific Language that our client company is using) projects

### Compilation and Configuration

Please see: [building-smallrace.md](./docs/building-smallrace/README.md) 

<br/>

---
<br/>

## Coderrect

Coderrect is an integrated tool that includes features such as code analysis, generating LLVM IR code, bug detection, generating reports, etc.

### Switch Target Languages

For the convenience of switching targe language, Coderrect organizes and distinguishes target languages using git branches. 

Commonly used branches include:

- [rust](https://github.com/coderrect-inc/coderrect/tree/rust): supports Rust language, for Solana projects
- [move](https://github.com/coderrect-inc/coderrect/tree/move): supports Solidity<sup>1</sup> language, for ETH projects
- [stx](https://github.com/coderrect-inc/coderrect/tree/stx): supports Smalltalk (a Domain Specific Language that our client company is using) projects

**NOTE** [1]: This naming may cause confusion. Solidity support JSON file is temporarily put in the `move` branch. 

_[Move language](https://github.com/move-language/move) is another smart contract language designed by Meta team. X-Ray tool's Move support is still WIP._


### Compilation and Execution

#### 1. Download repo and select target language

Download the repo and switch to the target language branch (we choose Rust here for Solana apps):
```
git clone https://github.com/coderrect-inc/coderrect
cd coderrect && git checkout rust
```

#### 2. Install

Navigate to the Go directory and directly run the existing `run` command:
```
cd gosrc
./run
```

**NOTE:** use `run` instead of `make all`, this script will copy the compiled binaries, `coderrect.json` and `data reporter` to the path at `whereis coderrect`.

The last line of the screen output should be:
```
Copying all binaries under [current-path]/coderrect/gosrc/bin to [coderrect-system-path]/bin
```

Note that only `coderrect.json` and the relevant binaries of coderrect are copied here. Binaries from smallrace and LLVMRace are not copied. You still need to manually create the link.

#### 3. Create Coderrect bin path and add to system $PATH

The `run` script locates the path using the `which coderrect` command.
```
CR=$(dirname $(which coderrect))
```

For first time running, `coderrect` binary doesn't exist. You can create a easy path, such as `~/local/sec3/bin/`, then generate a DUMMY coderrect binary:
```
mkdir [your-path]/local/sec3/bin/
touch [your-path]/local/sec3/bin/coderrect
chmod 755 [your-path]/local/sec3/bin/coderrect
```

You can run `which` or `whereis` to confirm the path:
```
$ whereis coderrect
coderrect: /home/your-username/local/sec3/bin/coderrect
```
Then add that path to `$PATH` and save it to `.bashrc`.
```
export PATH="[your-path]/local/sec3/bin:$PATH"
```
Then run the script `run` again, the new build `coderrect` binary will overwrite the dummy one. 

#### 4. create binary links for LLVMRace and smallrace

Create two soft links for `LLVMRace` and `smallrace` binaries, which should be put at the same path as `Coderrect`.

```
ln -s [your-project-path]/LLVMRace/build/bin/racedetect /home/your-username/local/sec3/bin/racedetect
ln -s [your-project-path]/smallrace/build/bin/sol-racedetect /home/your-username/local/sec3/bin/sol-racedetect
```

### Tips

For `smallrace` and `LLVMRace`, every time you switch branches, the **name of binary** generated for different target languages (such as Rust, Solidity, etc) will be different. 

If the new built binaries are not linked yet, you need to create new links for them. See [create binary links for LLVMRace and smallrace](./README.md#4-create-binary-links-for-llvmrace-and-smallrace)  

<br/>

Coderrect repo provided a script `buildpkg` for quick installation, compiling all dependency packages (including LLVMRace, etc.):
```
cd coderrect/installer/
./buildpkg -d
```
But this script is not well maintained and may have bugs. There may be issues with the software source, causing download speed slow. It is better to compile `LLVMRace` and `smallrace` separately.


<br/>
<br/>


## Addendum

WIP: suuport of LLVM 17 and later version. [doc](./docs/latest-LLVM-support.md). 



