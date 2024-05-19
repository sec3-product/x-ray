# Windows Building Instruction

## Before You Start
There are several things to make sure before you start building Coderrect Windows:
1. Upgrad your Windows 10 to **Window 10 Pro or Enterprise (or Education)**
2. Make sure you OS version is >= 10.0.18363.1500 (this requirement can be loosen, as long as we can run a based windows image that can install VS2019)
3. Install Docker ([link](https://docs.docker.com/docker-for-windows/install/))
4. After Docker is running, right click the Docker icon and select "Switch to windows container" (follow Docker's instructions to turn on Hyper-V and Container features)
5. Make sure to put all coderrect's git repo under `c:\code\`, including:
    1. `classic-flang-llvm-project`: customized LLVM
    2. `LLVMRace`: racedetect
    3. `coderrect`: everything else
6. Switch `LLVMRace` and `coderrect` to their `windows` branch.

## Step 1: Install Docker Image and Run Docker Container (Optional)
**NOTE:** You can skip this step if you already built the docker image and know how to run it.

```
cd C:\code\coderrect\installer\win
.\rundocker.bat
```

The script `rundocker.bat` will automatically build the coderrect windows docker image if it's not installed, and start a container using the image.

If you prefer to pull the pre-built docker image from docker hub. You can use the command below:

```
.\rundocker.bat -d
```

which will pull the docker image from docker hub and start a docker container based on it.


## Step 2: Build LLVM (Optional)
**NOTE:** You can skip this step if you alredy built LLVM using MinGW.

The reason for split step 2 from step 3 is due to the fact LLVM building time is too long.
Meanwhile, the changes in our LLVM repo is infrequent.
So we can manually trigger LLVM to build when necessary

```
cd C:\code\coderrect\installer\win
.\buildllvm.bat
```

By default, if this script find `C:\code\classic-flang-llvm-project` exists, it will build LLVM using `make -j8`.

If `C:\code\classic-flang-llvm-project` doesn't exist and no flag is specified, the script will download a pre-built llvm project to this location.

If you want to use more threads when building, you can pass the argument to `buildllvm.bat` (the following example will build with `make -j16`):

```
.\buildllvm.bat -j16
```

You can also explicitly ask the script to download the pre-built llvm project:

```
.\buildllvm.bat -d
```

**It's possible the pre-built llvm project is not up-to-date.**

## Step 3: Build Coderrect Windows

To build a develop installer:
```
cd C:\code\coderrect\installer\win
.\buildpkg -d
```

To build a release installer (for example, 1.0.0):
```
cd C:\code\coderrect\installer\win
.\buildpkg -r 1.0.0
```

The final installer will be placed at `C:\code\coderrect\installer`

## What's in the docker image
The docker image contains a building environment to build ALL components for coderrect windows package.

It uses [Chocolatey](https://chocolatey.org/) as the package manager.
All packages except Visual Studio 2019 are installed under `c:\ProgramData\chocolatey`.
Visual Studio 2019 is installed under `c:\Program Files (x86)\Microsoft Visual Studio`.

The image includes:
1. [MinGW 8.1.0](https://community.chocolatey.org/packages/mingw/8.1.0) (gcc/g++)
2. [cmake](https://community.chocolatey.org/packages/cmake)
3. [make](https://community.chocolatey.org/packages/make)
4. [golang](https://community.chocolatey.org/packages/golang)
5. [python3](https://community.chocolatey.org/packages/python/3.9.4): for building LLVM
6. [nsis3](https://community.chocolatey.org/packages/nsis): for creating Windows installation package
7. [visual studio 2019 community](https://community.chocolatey.org/packages/visualstudio2019community): for building Windows Libear
8. [Desktop development with C++ workload for Visual Studio 2019](https://community.chocolatey.org/packages/visualstudio2019-workload-nativedesktop): for building Windows Libear
9. [Git](https://community.chocolatey.org/packages/git)
10. [Vim](https://community.chocolatey.org/packages/vim)

In addition, two extensions of NSIS 3.06.1 are manually installed:
1. [NSIS advanced logging](https://nsis.sourceforge.io/Special_Builds): for log-based uninstallation
2. [EnVar Plugin](https://nsis.sourceforge.io/EnVar_plug-in): for acquiring environment variables

## Caveats
There are three dll files need to be included in the `bin\` folder of the built package. They are:
1. `libgcc_s_seh-1.dll`
2. `libstdc++-6.dll`
3. `libwinpthread-1.dll`

These dll files are directly copied from MinGW, which are placed under `C:\Progarm Files\mingw-x64\x86_64-8.1.0-posix-seh-rt_v6-rev\mingw64\bin`.
I've also uploaded a copy of them to the AWS server (check the `:DownloadDLLs` procedure in `buildpkg.bat`).

There are several other dll files in MinGW that could be useful for future, such as `libgomp-1.dll` (placed under the same path).

## Unsolved Issues:

### 1. Error Handling of Batch Scripts
Now our batch script does not have an error handling mechanism.
So building errors will not abort the whole building script.

This could make us to ignore some intermediate building erros.
