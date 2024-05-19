import pytest
import os
import re

expected_no_race = 'No race detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestTDengine:
    proj = "TDengine"
    buildDir = os.path.join(home, proj)
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf TDengine && \
                  git clone https://github.com/taosdata/TDengine.git")

    def teardown_class(self):
        os.system("cd " + home + " && \
                  rm -rf TDengine")
        
    @pytest.mark.timeout(120)
    def test_TDengine(self):
        r = os.system("cd " + home + " && \
                      cd TDengine && \
                      git submodule update --init --recursive && \
                      docker-coderrect -e taosd \"mkdir build;cd build;cmake ..;make -j $(nproc)\" > " + TestTDengine.logFile + " 2>&1")

        assert r == 0
        
        with open(TestTDengine.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
