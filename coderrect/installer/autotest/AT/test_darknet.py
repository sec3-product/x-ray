import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestDarknet:
    proj = "darknet"
    buildDir = os.path.join(home, proj)
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf darknet && \
                  git clone https://github.com/pjreddie/darknet.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestDarknet.logFile + " " + aws_report_dir)
            os.system("cd " + TestDarknet.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestDarknet.proj + ".log")

    @pytest.mark.timeout(60)
    def test_darknet(self):
        r = os.system("cd " + home + " && \
                      cd darknet && \
                      coderrect -e darknet make -j $(nproc) > " + TestDarknet.logFile + " 2>&1")

        assert r == 0

        f = open(TestDarknet.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
