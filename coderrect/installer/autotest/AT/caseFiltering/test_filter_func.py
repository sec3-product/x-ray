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
                  git clone https://github.com/memcached/memcached.git && \
                  cd memcached && \
                  ./autogen.sh && \
                  ./configure")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestMemcached.logFile + " " + aws_report_dir)
            os.system("cd " + TestMemcached.buildDir + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestMemcached.proj + ".log")

    @pytest.mark.timeout(60)
    def test_filtering(self):
        #check race number without custom config
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      coderrect -e memcached make -j $(nproc) > " + TestMemcached.logFile + " 2>&1")

        assert r == 0

        f = open(TestMemcached.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        baseNum = int(ret.group(1))

        #check race number with function filtering
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      make clean && \
                      coderrect -e memcached -conf=../mem_func.json make -j $(nproc) > " + TestMemcached.logFile + " 2>&1")

        assert r == 0

        f = open(TestMemcached.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) == baseNum - 1

        #check race number with variable filtering
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      make clean && \
                      coderrect -e memcached -conf=../mem_variable.json make -j $(nproc) > " + TestMemcached.logFile + " 2>&1")

        assert r == 0

        f = open(TestMemcached.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) == baseNum - 3

        #check race number with code line filtering
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      make clean && \
                      coderrect -e memcached -conf=../mem_line.json make -j $(nproc) > " + TestMemcached.logFile + " 2>&1")

        assert r == 0

        f = open(TestMemcached.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) == baseNum - 1
