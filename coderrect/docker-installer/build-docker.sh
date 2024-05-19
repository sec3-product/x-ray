#!/bin/bash

printHelp() {
  cat << EOF
Build docker image using coderrect scan.

Usage:
  docker-coderrect [options] build-command

  -h            
    Print this message 
  -c, --hpc-version
    Set the HPC coderrect version to download
  -s, --system-version
    Download all specific version of coderrect
  -l, --local-file
    Use local versioned coderrect

Examples:
  1. Download developer coderrect
  \$ build-docker.sh

  2. Download HPC specified version of coderrect
  \$ build-docker.sh -c 0.9.2

  3. Add local coderrect to image with system tag
  \$ build-docker.sh -s 0.9.2 -l /path/to/coderrect
EOF
}

pkgName=develop
version=
localPath=
dFile=
while (( "$#" )); do
    case "$1" in
        -h|--help)
            printHelp
            exit 3
            ;;
        -c|--hpc-version)
            version=$2
            pkgName=hpc
            shift 2
            ;;
        -s|--system-version)
            version=$2
            pkgName=system
            shift 2
            ;;
        -l|--local-file)
            localPath=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            printHelp
            exit 3
            ;;
    esac
done

# Verify parameters
if [ -z "$version" ] ; then
    coderrectVersion=$pkgName
else
    coderrectVersion=$pkgName-$version
fi 

# Download from official web site
if [ -z "$localPath" ] ; then
    dFileName=coderrect-linux-${coderrectVersion}.tar.gz
    if [ -f "${dFileName}" ]; then
        echo "$dFileName exist, delete it"
        rm ${dFileName}
    else 
        echo "$FILE does not exist."
    fi
    wget --no-check-certificate https://public-installer-pkg.s3.us-east-2.amazonaws.com/coderrect-linux-${coderrectVersion}.tar.gz
    localPath=coderrect-linux-${coderrectVersion}.tar.gz
fi 

# call docker build
localFileName=$(basename $localPath .tar.gz)
echo "building docker image with local file ${localPath}, ${localFileName}"
docker build --build-arg CODERRECT_LOCAL_PATH=$localPath --build-arg CODERRECT_LOCAL_FILENAME=$localFileName -t coderrect/coderrect:linux-$coderrectVersion .
if [ "$pkgName" = "hpc" ]; then
    docker tag coderrect/coderrect:linux-$coderrectVersion coderrect/coderrect:latest
fi

# clean
if [ ! -z "$dFileName" ] ; then
    rm $dFileName
fi