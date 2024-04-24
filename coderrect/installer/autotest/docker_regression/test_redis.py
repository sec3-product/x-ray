import pytest
import os
import re

expected_no_race = 'No race detected.'
expected_race = '\\s+(\\d+) shared data races'
home = os.path.dirname(os.path.realpath(__file__))

class TestRedis:
    proj = "redis"
    buildDir = os.path.join(home, proj)
    logFile = buildDir + "/at_" + proj + ".log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf redis && \
                  git clone https://github.com/redis/redis.git")

    def teardown_class(self):
        os.system("cd " + home + " && \
                  rm -rf redis")
        
    @pytest.mark.timeout(210)
    def test_TDengine(self):
        r = os.system("cd " + home + " && \
                      cd redis && \
                      git checkout 6.0.0 && \
                      docker-coderrect -e redis-server make -j $(nproc) > " + TestRedis.logFile + " 2>&1")

        assert r == 0
        
        with open(TestRedis.logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
