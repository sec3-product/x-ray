import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

proj = "stdlib"
buildDir = os.path.join(home, proj)
logFile = buildDir + "/at_" + proj + ".log"

@pytest.mark.skip(reason='running long time without race found')
class TestStdlib:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf stdlib && \
                  git clone https://github.com/fortran-lang/stdlib.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + logFile + " " + aws_report_dir)
            os.system("cd " + buildDir + " && \
                      cp build/.coderrect/logs/log.current " + aws_report_dir + "/log_current_" + proj + ".log")

    @pytest.mark.timeout(60)
    def test_stdlib(self):
        r = os.system("cd " + home + " && \
                      cd stdlib && \
                      mkdir build && \
                      cd build && \
                      pip3 install fypp && \
                      cmake .. && \
                      coderrect -e test_varn,test_mean,test_linalg make -j $(nproc) > " + logFile + " 2>&1")

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t
