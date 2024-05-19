#!/bin/bash

summary=test_summary
export PATH=$CODERRECT_HOME/bin:$PATH

rm -rf /installer/build/test

test_dir=/installer/build/test
mkdir -p $test_dir/report

cp -r $CODERRECT_HOME/examples $test_dir
cp /installer/autotest/parseTest.py $test_dir
cp /installer/autotest/compareBenchmark.py $test_dir/report
cp /installer/autotest/benchmark.track $test_dir/report

cd $test_dir

#delete fortran cases if regression on system package.
if [[ $CODERRECT_HOME =~ coderrect-linux-system ]];then
	#rm -rf AT/fortran
	rm -rf examples/fortran
fi

pytest -n auto 2>&1 | tee $summary
