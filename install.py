#!/usr/bin/env python3

import os
import shutil
import subprocess
import getopt
import sys
from datetime import datetime

def pull_components(release, develop, rust):
   # get args passed in
   args = (release, develop, rust)

   # we will build our package under build
   if not os.path.exists("build"):
       os.mkdir("build")
       print("Create build directory")
   print("Build directory created")
   
   # llvm-project
   # clone llvm-project
   if not os.path.exists("build/llvm12"):
       subprocess.run("cd build && git clone --recursive ssh://git@github.com/coderrect-inc/llvm12.git", shell=True, check=True)
   # pull llvm-project
   else:
       subprocess.run("cd build/llvm12 && git pull --all", shell=True, check=True)
   print("Finish updating llvm12")

###########################################
#                  docker                 #
###########################################

def build_in_docker(release, develop, rust):
   # get args passed in
   args = (release, develop, rust)
   # check if docker image has been installed
   result = subprocess.run("docker images | grep 'coderrect/installer' | grep 1.2 | wc -l", shell=True, capture_output=True, text=True)
   if result.stdout.strip() != "1":
       print("Docker image coderrect/installer:1.2 not installed, please run build-coderrect-installer-1.1-docker.sh first", file=sys.stderr)
       sys.exit(1)

   print("Start building all coderrect's components in the docker container...")
   # run a container to do all builds
   if rust:
       subprocess.run("docker run --rm -v $(pwd)/build:/build -v $(pwd)/dockerstuff/scripts:/scripts --user=$(id -u):$(id -g) coderrect/installer:1.2 /scripts/build-components-llvm12-rust.sh", shell=True, check=True)
   else:
       subprocess.run("docker run --rm -v $(pwd)/build:/build -v $(pwd)/dockerstuff/scripts:/scripts --user=$(id -u):$(id -g) coderrect/installer:1.1 /scripts/build-components.sh", shell=True, check=True)

def build_package(release, develop, rust):
   # get args passed in
   args = (release, develop, rust)
   version = ""
   # get version type
   if release:
       version = pkg_version
   elif rust:
       version = "solana"
   
   # get package timestamp
   build = subprocess.run("date +'%s'", shell=True, capture_output=True, text=True).stdout.strip()

   print("Start copying files to build/package...")
   # `rm -rf build/package && mkdir -p build/package`;
   if not os.path.exists("build/package"):
       os.makedirs("build/package")
   with open("build/package/VERSION", "w") as f:
       f.write(f"package {version} build {build}")
   
   # copy things to package/bin
   if not os.path.exists("build/package/bin"):
       os.makedirs("build/package/bin")
   subprocess.run("cp build/coderrect/gosrc/bin/* build/package/bin/", shell=True, check=True)
   subprocess.run("cp build/coderrect/coderrect-exec/coderrect-exec build/package/bin/", shell=True, check=True)
   subprocess.run("cp build/coderrect/libear/build/libear/libear.so build/package/bin/", shell=True, check=True)
   subprocess.run("cp build/coderrect/docker-installer/docker-coderrect build/package/bin/", shell=True, check=True)
   subprocess.run("cp -r build/LLVMRace/build/bin/* build/package/bin/", shell=True, check=True)
   
   # copy things to package/clang
   if not os.path.exists("build/package/clang"):
       os.makedirs("build/package/clang/bin")
       os.makedirs("build/package/clang/include")
       os.makedirs("build/package/clang/lib")
   
   if rust:
       subprocess.run("cp build/smallrace/build/bin/sol-racedetect build/package/bin/", shell=True, check=True)
       os.chdir("build/package/bin")
       os.symlink("coderrect", "soteria")
       os.chdir("../../..")

   else:
       subprocess.run("cp build/custom-clang/bin/clang-10 build/package/clang/bin/", shell=True, check=True)
       subprocess.run("cp build/custom-clang/bin/flang1 build/package/clang/bin/", shell=True, check=True)
       subprocess.run("cp build/custom-clang/bin/flang2 build/package/clang/bin/", shell=True, check=True)
       os.chdir("build/package/clang/bin")
       os.symlink("clang-10", "clang")
       os.symlink("clang-10", "clang++")
       os.symlink("clang-10", "flang")
       os.chdir("../../..")
   
   subprocess.run("cp build/custom-clang/bin/llvm-link build/package/clang/bin/", shell=True, check=True)
   subprocess.run("cp build/custom-clang/lib/libomp.so build/package/bin/", shell=True, check=True)
   subprocess.run("cp -r build/custom-clang/include/* build/package/clang/include", shell=True, check=True)
   subprocess.run("cp -r build/custom-clang/lib/clang build/package/clang/lib", shell=True, check=True)
   subprocess.run("cp -r build/custom-clang/lib/cmake build/package/clang/lib", shell=True, check=True)
   
   # copy things to package/conf, package/data, and package/EULA.txt
   subprocess.run("cp -r build/coderrect/installer/package/* build/package/", shell=True, check=True)
   
   print("Finish copying files to build/package...")

   # create tarball based on the build type
   print("Start creating a tarball for build/package...")
   if develop:
       shutil.rmtree("build/coderrect-linux-develop", ignore_errors=True)
       shutil.move("build/package", "build/coderrect-linux-develop")
       os.chdir("build")
       subprocess.run("tar -czvf coderrect-linux-develop.tar.gz coderrect-linux-develop", shell=True, check=True)
       os.chdir("..")
       print("Final package is created at build/coderrect-linux-develop.tar.gz")
   elif rust:
       shutil.rmtree("build/soteria-linux-rust", ignore_errors=True)
       shutil.move("build/package", "build/soteria-linux-rust")
       os.chdir("build")
       subprocess.run("tar -czvf soteria-linux-rust.tar.gz soteria-linux-rust", shell=True, check=True)
       os.chdir("..")
       print("Final package is created at build/soteria-linux-rust.tar.gz")
   elif release:
       shutil.rmtree(f"build/coderrect-linux-{pkg_version}", ignore_errors=True)
       shutil.move("build/package", f"build/coderrect-linux-{pkg_version}")
       os.chdir("build")
       subprocess.run(f"tar -czvf coderrect-linux-{pkg_version}.tar.gz coderrect-linux-{pkg_version}", shell=True, check=True)
       os.chdir("..")
       print(f"Final package is created at build/coderrect-linux-{pkg_version}.tar.gz")
   else:
       print("No build type specified, no tarball created. Use either -d, -s, -lam, -r <version number>, -x, -m, or -e to specify the build type.", file=sys.stderr)
       sys.exit(1)


###########################################
#                  utils                  #
###########################################

import os
import subprocess


# modules to compile using Go
def compile_coderrect_go(folder):
    print(f"Compiling {folder} with Go ...")
    os.chdir(folder)
    subprocess.run(["cd", "gosrc"], check=True)
    subprocess.run(["./run"], check=True)
    os.chdir("..")

# modules to compile using CMake
def compile_parser_cmake(folder):
    print(f"Compiling {folder} with CMake...")
    os.chdir(folder)
    subprocess.run(["mkdir", "build"], check=True)
    subprocess.run(["cd", "build"], check=True)
    subprocess.run(["cmake", ".."], check=True)
    subprocess.run(["make"], check=True)
    os.chdir("..")

def compile_scanner_cmake(folder):
    print(f"Compiling {folder} with CMake...")
    os.chdir(folder)
    subprocess.run(["mkdir", "build"], check=True)
    subprocess.run(["cd", "build"], check=True)
    subprocess.run(["cmake", ".."], check=True)
    subprocess.run(["make"], check=True)
    os.chdir("..")

# add compiled LLVMRace tool to PATH
def add_to_path(directory):
    # Get the current PATH environment variable
    current_path = os.getenv('PATH', '')

    # If the directory is already in the PATH, do nothing
    if directory in current_path.split(os.pathsep):
        print(f"{directory} is already in the PATH")
        return

    # Add the directory to the PATH
    os.environ['PATH'] = f"{current_path}{os.pathsep}{directory}"
    print(f"{directory} has been added to the PATH")


###########################################
#                  main                   #
###########################################

# TODO: LLVMRace binary
LLVMRace_binary = ''
add_to_path(LLVMRace)
    
# TODO: pull LLVM 12 and compile


# Compile each subdirectory
compile_coderrect_go("coderrect")
compile_parser_cmake("code-parser")
compile_scanner_cmake("code-scanner")

# Check Go version
print("Checking Go version...")
go_version = subprocess.check_output(["go", "version"], text=True)
print(f"Go version: {go_version}")

# TODO: check CMake version

# pull_components(release, develop, rust)
# build_in_docker(release, develop, rust)
# build_package(release, develop, rust)