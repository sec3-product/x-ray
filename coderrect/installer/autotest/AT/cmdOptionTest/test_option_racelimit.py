import os
import re
import json
import pytest

home = os.path.dirname(os.path.realpath(__file__))
tmpFile = home + "/test_option_racelimit/tmp.file"

class Test_RaceLimit:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf test_option_racelimit && \
                  mkdir test_option_racelimit && \
                  cd test_option_racelimit && \
                  git clone https://github.com/pjreddie/darknet.git")
    
    def test_racelimit(self):
        #run without config file
        r = os.system("cd " + home + " && \
                      cd test_option_racelimit/darknet && \
                      coderrect -e darknet make -j $(nproc) > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('(\\d+) shared data races', t)
        assert ret is not None
        assert int(ret.group(1)) > 1

        #run with config file
        r = os.system("cd " + home + " && \
                      cd test_option_racelimit/darknet && \
                      make clean && \
                      coderrect -e darknet -conf=../../raceLimit.json make -j $(nproc) > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('(\\d+) shared data races', t)
        assert ret is not None
        assert int(ret.group(1)) == 1
