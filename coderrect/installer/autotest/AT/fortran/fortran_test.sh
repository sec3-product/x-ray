#!/bin/bash

function usage() {
	echo "$0 		     #use default pkg on \$CODERRECT_HOME"
	echo "$0 coderrect_pkg_path  #customize the absolute path of coderrect pkg."
}

fortran_dir=$(cd "$(dirname "$0")"; pwd)

#get location of coderrect pkg, check command line firstly, then system env.
if [ $# -eq 0 ];then
	if [ $CODERRECT_HOME ];then
		coderrect_abs_path=$CODERRECT_HOME
	else
		usage
		exit 1
	fi
else
	coderrect_abs_path=$1
fi

docker run --rm \
	   -v $fortran_dir:/fortran_test \
	   -v $coderrect_abs_path:/coderrect \
	   -e CODERRECT_HOME=/coderrect \
	   coderrect/testbed:1.0 \
	   sh -c 'export PATH=$CODERRECT_HOME/bin:$PATH && cd /fortran_test && pytest -n auto'
