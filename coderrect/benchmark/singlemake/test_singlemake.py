import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = 'detected (\\d+) races in total.'
home = os.path.dirname(os.path.realpath(__file__))

class TestSingleMake:
    proj = "singlemake"
    buildDir = home
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        pass

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestSingleMake.logFile + " " + aws_report_dir)
            os.system("cd " + TestSingleMake.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestSingleMake.proj + ".log")

    #@pytest.mark.timeout(27)
    def test_singlemake(self):
        r = os.system("cd " + TestSingleMake.buildDir + " && \
                      coderrect make > " + TestSingleMake.logFile + " 2>&1")

        assert r == 0

        with open(TestSingleMake.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t
