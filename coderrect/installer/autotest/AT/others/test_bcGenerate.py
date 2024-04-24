import pytest
import os
import re

expected_no_race = 'No race detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestMemcached:
    proj = "memcached"
    buildDir = os.path.join(home, proj)
    tmpFile = buildDir + "/tmp.log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf memcached && \
                  git clone https://github.com/memcached/memcached.git")

    def teardown_class(self):
        pass

    @pytest.mark.timeout(60)
    def test_memcached(self):
        r = os.system("cd " + home + " && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      coderrect -e memcached make -j $(nproc)")

        assert r == 0
        
        r = os.system("cd " + home + " && \
                  cd memcached && \
                  find . -name *.bc -print | grep -v .o.bc > " + TestMemcached.tmpFile)
        assert r == 0

        f = open(TestMemcached.tmpFile, 'r', encoding='utf-8')
        t = f.readlines()
        f.close()
        assert len(t) == 1
