import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

class TestDRB001:
    proj = "DRB001"
    buildDir = home + "/test_DRB0001"
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf test_DRB0001 && \
                  mkdir test_DRB0001")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestDRB001.logFile + " " + aws_report_dir)
            os.system("cd " + TestDRB001.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestDRB001.proj + ".log")

    #@pytest.mark.timeout(27)
    def test_helloworld(self):
        r = os.system("cd " + TestDRB001.buildDir + " && \
                      coderrect gfortran -fopenmp ../DRB001-antidep1-orig-yes.f95 > " + TestDRB001.logFile + " 2>&1")

        assert r == 0

        with open(TestDRB001.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
