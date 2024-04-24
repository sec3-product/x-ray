import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = 'detected (\\d+) races in total.'
home = os.path.dirname(os.path.realpath(__file__))

class TestHelloc:
    proj = "hello_c"
    buildDir = home
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        pass

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestHelloc.logFile + " " + aws_report_dir)
            os.system("cd " + TestHelloc.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestHelloc.proj + ".log")

    #@pytest.mark.timeout(27)
    def test_hello_c(self):
        r = os.system("cd " + TestHelloc.buildDir + " && \
                      coderrect clang -fopenmp -g hello.c > " + TestHelloc.logFile + " 2>&1")

        assert r == 0

        with open(TestHelloc.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t
