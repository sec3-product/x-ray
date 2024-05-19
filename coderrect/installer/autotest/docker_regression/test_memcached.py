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
        os.system("cd " + home + " && \
                  rm -rf memcached")

    @pytest.mark.timeout(60)
    def test_memcached(self):
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      docker-coderrect -e memcached make -j $(nproc) > " + TestMemcached.logFile + " 2>&1")

        assert r == 0

        with open(TestMemcached.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
