import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestCblosc:
    proj = "c-blosc"
    buildDir = os.path.join(home, proj, 'build')
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf c-blosc && \
                  git clone https://github.com/Blosc/c-blosc.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestCblosc.logFile + " " + aws_report_dir)
            os.system("cd " + TestCblosc.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestCblosc.proj + ".log")

    #@pytest.mark.timeout(60)
    def test_cblosc(self):
        r = os.system("cd " + home + " && \
                      cd c-blosc && \
                      git checkout v1.17.1 && \
                      mkdir build && \
                      cd build && \
                      cmake .. > " + TestCblosc.logFile + " 2>&1 && \
                      coderrect -e test_nthreads make >> " + TestCblosc.logFile + " 2>&1")

        assert r == 0

        f = open(TestCblosc.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
