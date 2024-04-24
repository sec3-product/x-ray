#!/bin/bash

installer_full_path=$(pwd)
coderrect_pkg=$1
images=$(docker images | grep coderrect/compat | awk '{print $1}')

function usage() {
	echo "Usage:   $0 package  #Name of coderrect package needed."
	echo "Example: $0 coderrect-linux-develop"
}

if [ $# -lt 1 ];then
	usage
	exit 1
fi

for i in $images;
do
	echo "---------compatible testing with image $i ---------" 

	docker run --rm \
	   	   --user=$(id -u):$(id -g) \
	   	   -v $installer_full_path:/installer \
	   	   -e CODERRECT_HOME=/installer/$coderrect_pkg \
	   	   $i \
	   	   /installer/autotest/run_compat.sh

	#print bencmark comparing result.
	`cd build/test/report && python3 compareBenchmark.py benchmark.track at_benchmark.log`
done
