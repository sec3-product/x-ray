import pytest
import os
import re

expected_no_race = 'No race detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestMemcached:
    proj = "memcached"
    buildDir = os.path.join(home, proj)
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf memcached && \
                  git clone https://github.com/memcached/memcached.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestMemcached.logFile + " " + aws_report_dir)
            os.system("cd " + TestMemcached.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestMemcached.proj + ".log")

    @pytest.mark.timeout(60)
    def test_memcached(self):
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      coderrect -e memcached make -j $(nproc) > " + TestMemcached.logFile + " 2>&1")

        assert r == 0

        f = open(TestMemcached.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
