# Building

Run the following from the LLVMRace root directory

```bash
LLVM_DIR=/home/jeff/llvm12/llvm-project/build/lib/cmake/llvm/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

# Running

To run the tool, simply type 
```console
./bin/racedetect /your/path/to/the/IR/file
```

## (Optional) VSCode Extension

Install the `Remote - Containers` extenstion by Microsoft.
Open the project in VSCode.
Click the Dialog that says "Open in container".
Wait for VSCode to open in the container. This may take a few minutes the first time.

### VSCode Build Task
Once VSCode has started in the container, the project can be built using the predefined build task.

The build task can be run by pressing `Shift+Cmd+B` on a mac, or by pressing `Shift+Cmd+P` to open the commadn pallete and selecting "Tasks: Run Build Task".

### Manual Build
The project can also be built manually by running the following from the LLVMRace root directory.

```bash
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang ..
```


### Building LLVM Locally

```bash
git clone https://github.com/llvm/llvm-project.git
mkdir -p llvm-project/build
cd llvm-project/build
git checkout release/12.x
cmake -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt;lld;mlir" ../llvm/ -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
