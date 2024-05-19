import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestCurl:
    proj = "curl"
    buildDir = os.path.join(home, proj, 'build')
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf curl && \
                  git clone https://github.com/curl/curl.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestCurl.logFile + " " + aws_report_dir)
            os.system("cd " + TestCurl.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestCurl.proj + ".log")

    #@pytest.mark.timeout(75)
    def test_curl(self):
        r = os.system("cd " + home + " && \
                      cd curl && \
                      git checkout curl-7_69_0 && \
                      mkdir build && \
                      cd build && \
                      cmake .. > " + TestCurl.logFile + " 2>&1 && \
                      coderrect -e lib1565 make >> " + TestCurl.logFile + " 2>&1")

        assert r == 0

        f = open(TestCurl.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
