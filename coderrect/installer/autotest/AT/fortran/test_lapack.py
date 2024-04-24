import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

proj = "lapack"
buildDir = os.path.join(home, proj)
logFile = buildDir + "/at_" + proj + ".log"

class TestLapack:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf lapack && \
                  git clone https://github.com/Reference-LAPACK/lapack.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + logFile + " " + aws_report_dir)
            os.system("cd " + buildDir + " && \
                      cp build/.coderrect/logs/log.current " + aws_report_dir + "/log_current_" + proj + ".log")

    @pytest.mark.timeout(150)
    def test_lapack(self):
        r = os.system("cd " + home + " && \
                      cd lapack && \
                      mkdir build && \
                      cd build && \
                      cmake .. && \
                      coderrect -e liblapack.a -conf=../../lapack.json make -j $(nproc) > " + logFile + " 2>&1")

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
