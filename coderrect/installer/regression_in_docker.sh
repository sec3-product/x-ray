#Pre-requirement:
#  1.Docker image coderrect/testbed:1.0 is needed in the script running.
#    You may create the image by type:
#    $ cd installer/dockerstuff
#    $ docker build -f Dockerfile.testbed -t coderrect/testbed:1.0 .
#
#  2.Under installer directory you need have a coderrect package
#    which is also the parameter specified in command line.

#!/bin/bash

installer_full_path=$(pwd)
coderrect_pkg=$1

function usage() {
	echo "Usage:   $0 package  #Name of coderrect package needed."
	echo "Example: $0 coderrect-linux-develop"
}

if [ $# -lt 1 ];then
	usage
	exit 1
else
	echo "Run regression testing with package: $1."
fi 

docker run --rm \
	   --user=$(id -u):$(id -g) \
	   -v $installer_full_path:/installer \
	   -v /home/cicd/repos:/home/ubuntu/repos \
	   -e CODERRECT_HOME=/installer/$coderrect_pkg \
	   coderrect/testbed \
	   /installer/autotest/run_regression.sh
