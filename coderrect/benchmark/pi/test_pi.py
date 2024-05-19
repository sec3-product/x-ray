import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

class TestPi:
    proj = "pi"
    buildDir = home
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        pass

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestPi.logFile + " " + aws_report_dir)
            os.system("cd " + TestPi.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestPi.proj + ".log")

    #@pytest.mark.timeout(27)
    def test_pi(self):
        r = os.system("cd " + TestPi.buildDir + " && \
                      coderrect clang -fopenmp -g pi.c > " + TestPi.logFile + " 2>&1")

        assert r == 0

        with open(TestPi.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
