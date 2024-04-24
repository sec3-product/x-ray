import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

class TestLulesh:
    proj = "lulesh"
    buildDir = os.path.join(home, "LULESH-2.0.3")
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf LULESH-2.0.3 && \
                  wget https://github.com/LLNL/LULESH/archive/2.0.3.zip && \
                  unzip 2.0.3.zip")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestLulesh.logFile + " " + aws_report_dir)
            os.system("cd " + TestLulesh.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestLulesh.proj + ".log")

    @pytest.mark.timeout(60)
    def test_lulesh(self):
        r = os.system("cd " + home + " && \
                      cd LULESH-2.0.3 && \
                      sed -i 's/CXX = $(MPICXX)/CXX = $(SERCXX)/g' Makefile && \
                      coderrect make -j $(nproc) > " + TestLulesh.logFile + " 2>&1")

        assert r == 0

        f = open(TestLulesh.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_no_race, t)
        assert ret is not None
