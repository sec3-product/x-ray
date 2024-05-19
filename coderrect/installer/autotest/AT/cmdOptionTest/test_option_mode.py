import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))
coderrect_home = os.getenv('CODERRECT_HOME')
pi_home = coderrect_home + "/examples/pi"

class Test_Mode:
    def test_fast(self):
        tmpFile = home + '/test_fast/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_fast && \
                      mkdir test_fast && \
                      cd test_fast && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -mode=fast clang -fopenmp -g pi.c > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('(\\d+) OpenMP races', t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    def test_exhaust(self):
        tmpFile = home + '/test_exhaust/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_exhaust && \
                      mkdir test_exhaust && \
                      cd test_exhaust && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -mode=exhaust clang -fopenmp -g pi.c > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('(\\d+) OpenMP races', t)
        assert ret is not None
        assert int(ret.group(1)) > 0
