import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestPbzip2:
    proj = "pbzip2"
    buildDir = os.path.join(home, 'sctbench/benchmarks/conc-bugs/pbzip2-0.9.4/pbzip2-0.9.4')
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf sctbench && \
                  git clone https://github.com/mc-imperial/sctbench.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestPbzip2.logFile + " " + aws_report_dir)
            os.system("cd " + TestPbzip2.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestPbzip2.proj + ".log")

    @pytest.mark.timeout(120)
    def test_pbzip2(self):
        r = os.system("cd " + home + " && \
                      cd sctbench/benchmarks/conc-bugs/pbzip2-0.9.4/pbzip2-0.9.4 && \
                      coderrect make -j $(nproc) > " + TestPbzip2.logFile + " 2>&1")

        assert r == 0

        f = open(TestPbzip2.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
