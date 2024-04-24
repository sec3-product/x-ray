import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) OpenMP races'
home = os.path.dirname(os.path.realpath(__file__))

class TestMiniAMR:
    proj = "miniAMR"
    buildDir = os.path.join(home, 'miniAMR-1.5.0/openmp')
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf miniAMR-1.5.0 && \
                  wget https://github.com/Mantevo/miniAMR/archive/v1.5.0.zip && \
                  unzip v1.5.0.zip")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestMiniAMR.logFile + " " + aws_report_dir)
            os.system("cd " + TestMiniAMR.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestMiniAMR.proj + ".log")

    @pytest.mark.timeout(60)
    def test_miniAMR(self):
        r = os.system("cd " + home + " && \
                      cd miniAMR-1.5.0/openmp && \
                      echo 'CC=mpicc' >> Makefile && \
                      echo 'LD=mpicc' >> Makefile && \
                      coderrect -e ma.x make -j $(nproc) > " + TestMiniAMR.logFile + " 2>&1")

        assert r == 0

        f = open(TestMiniAMR.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
