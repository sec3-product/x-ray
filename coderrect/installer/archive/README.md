# Build Instruction

### First Time Build
To build the Coderrect package from a fresh environment for the first time, you need to first build several docker images for later use.

- Build `coderrect/installer:1.0` using `dockerstuff/Dockerfile`
- Build `coderrect/installer:1.1` using `build-coderrect-installer-1.1-docker.sh`
- Build `coderrect/clang:1.0` using `dockerstff/Dockerfile.clang`
- Build `coderrect/flang:1.0` using `dockerstff/Dockerfile.fortran`

The docker build command should be `docker build -f <Dockerfile path> -t <image name:version> .`

### Build Dev Package and Upload

You can use the follow script to easily build our develop package and upload it to s3 server

```
!/bin/bash

cd coderrect
git pull
cd installer
./build-pkg -d
./upload-pkg coderrect-linux-develop.tar.gz
```

### Build Release Package and Upload

1. Create release branch for all of our components (coderrect and LLVMRace). Name them like `release-vx.x.x` where `x.x.x` is the version number.
2. `./build-package -r x.x.x`
4. `./upload-pkg coderrect-linux-system-x.x.x.tar.gz` and `./upload-pkg coderrect-linux-hpc-x.x.x.tar.gz`

## Remaining Issue:

At some point we used clang-11 and then switched back to clang-10.
It seems clang-11 is no longer needed for the whole package (need confirm from others).
Yet our build script still has a check for clang-11 (temporarily commented out).
