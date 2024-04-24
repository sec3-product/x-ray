# Building LLVMRace

### 1. Download repo and select target language

Download the repo and switch to the target language branch (we choose Rust here for Solana apps):
```
git clone https://github.com/coderrect-inc/LLVMRace
cd LLVMRace && git checkout rust
```

### 2. Set LLVM_DIR and MLIR_DIR path

Before running cmake, you need to modify `CMakeLists.txt` to specify the LLVM_DIR path.

On `sushi` server, there's a compiled repositories ready-to-use, just add two lines in CMakeLists.txt:
```
set(LLVM_DIR /home/jeff/llvm12/llvm-project/build/lib/cmake/llvm/)
set(MLIR_DIR /home/jeff/llvm12/llvm-project/build/lib/cmake/mlir/)
```
Otherwise, add your own built LLVM repo path.

### 3. Install
```
mkdir build && cd build
cmake ..
make -j$(nproc)
```
Simply use `make` instead of `make -j xx` if you don't need multi-thread.

<br/>

## Troubleshooting

#### z3 repo not found

Solution: Modify `LLVMRace/tools/RaceDetector/CMakeLists.txt`, add this line:
```
GIT_REPOSITORY "https://github.com/Z3Prover/z3"
```
**NOTE**: the above line should be removed after first-time building, otherwise it will slowdown the compilation process.
