# Summary
AT directory contains cases for regression testing, you may run it on docker or non-docker env, below have more details.

# Run in docker env (Recommend)
    
    //build docker image
    $ cd /home/xiaohua
    $ git clone git@github.com:coderrect-inc/installer.git
    $ cd installer/dockerstuff
    $ docker build -f Dockerfile.testbed -t coderrect/testbed:1.0 .
    
    //run regression
    $ cd /home/xiaohua/installer
    $ wget https://public-installer-pkg.s3.us-east-2.amazonaws.com/coderrect-linux-hpc-0.9.1.tar.gz
    $ tar -xzvf coderrect-linux-hpc-0.9.1.tar.gz
    $ docker run --rm -v /home/xiaohua/installer:/installer -e CODERRECT_HOME=/installer/coderrect-linux-hpc-0.9.1 coderrect/testbed:1.0 /installer/autotest/run_regression.sh
                 
    //Note: run_regression.sh will run cases under directory 'installer/autotest/AT' and 'coderrect-linux-xxx/examples' 
            together.

# Run in non-docker env
# Prereqirement
1.pytest and pytest plugin should be installed.

    $ pip3 install pytest pytest-timeout pytest-repeat pytest-xdist

2.install openMP on your testbed.

    $ sudo apt install libomp-dev

3.download coderrect package and configure env.
  
    $ wget https://public-installer-pkg.s3.us-east-2.amazonaws.com/coderrect-linux-develop.tar.gz
    $ tar -xzvf coderrect-linux-develop.tar.gz
    $ export CODERRECT_HOME=$PWD/coderrect-linux-develop
    $ export PATH=$CODERRECT_HOME/bin:$PATH

# Test
1.Run all cases paranelly under the directory including sub-directories

    $ cp -r installer/autotest/AT /tmp/test/
    $ cd /tmp/test/AT 
    $ pytest -n auto

2.Run single script

    $ pytest script-name.py
