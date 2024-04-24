import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

proj = "CloverLeaf_OpenMP"
buildDir = os.path.join(home, proj)
logFile = buildDir + "/at_" + proj + ".log"

class TestLapack:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf CloverLeaf_OpenMP && \
                  git clone https://github.com/UK-MAC/CloverLeaf_OpenMP.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + logFile + " " + aws_report_dir)
            os.system("cd " + buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + proj + ".log")

    @pytest.mark.timeout(120)
    def test_cloverleaf(self):
        r = os.system("cd " + buildDir + " && \
                      coderrect make COMPILER=GNU MPI_COMPILER=gfortran C_MPI_COMPILER=gcc > " + logFile + " 2>&1")

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
