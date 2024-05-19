import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = 'detected (\\d+) races in total.'
home = os.path.dirname(os.path.realpath(__file__))

class TestHelloworld:
    proj = "helloworld"
    buildDir = home + "/test_helloworld"
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf test_helloworld && \
                  mkdir test_helloworld")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestHelloworld.logFile + " " + aws_report_dir)
            os.system("cd " + TestHelloworld.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestHelloworld.proj + ".log")

    #@pytest.mark.timeout(27)
    def test_helloworld(self):
        r = os.system("cd " + TestHelloworld.buildDir + " && \
                      coderrect gfortran -lomp ../helloworld.f > " + TestHelloworld.logFile + " 2>&1")

        assert r == 0

        with open(TestHelloworld.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t
