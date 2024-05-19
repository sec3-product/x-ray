import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

proj = "moleculardynamics"
buildDir = os.path.join(home, proj)
logFile = buildDir + "/at_" + proj + ".log"

class TestMoleculardynamics:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf moleculardynamics && \
                  git clone https://github.com/alexpacheco/moleculardynamics.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + logFile + " " + aws_report_dir)
            os.system("cd " + buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + proj + ".log")

    @pytest.mark.timeout(60)
    def test_moleculardynamics(self):
        r = os.system("cd " + home + " && \
                      cd moleculardynamics && \
                      coderrect -e mdo make > " + logFile + " 2>&1")

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t
