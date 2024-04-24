import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

class TestQuicksilver:
    proj = "quicksilver"
    buildDir = os.path.join(home, 'Quicksilver/src')
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf Quicksilver && \
                  git clone https://github.com/LLNL/Quicksilver.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestQuicksilver.logFile + " " + aws_report_dir)
            os.system("cd " + TestQuicksilver.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestQuicksilver.proj + ".log")

    @pytest.mark.timeout(60)
    def test_quicksilver(self):
        r = os.system("cd " + home + " && \
                      cd Quicksilver/src && \
                      echo 'CXX = clang++' >> Makefile && \
                      echo 'CXXFLAGS = -std=c++14 -g ' >> Makefile && \
                      echo 'CPPFLAGS = -DHAVE_OPENMP -fopenmp -g' >> Makefile && \
                      echo 'LDFLAGS = -fopenmp' >> Makefile && \
                      coderrect make -j $(nproc) > " + TestQuicksilver.logFile + " 2>&1")

        assert r == 0

        f = open(TestQuicksilver.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
