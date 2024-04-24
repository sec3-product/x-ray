import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_Showconf:
    def test_option_showconf(self):
        tmpFile = home + '/test_option_showconf/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_option_showconf && \
                      mkdir test_option_showconf && \
                      cd test_option_showconf && \
                      coderrect -showconf > " + tmpFile)
        assert r == 0
        j = json.load(open(tmpFile))

        assert 'logger' in j.keys()
        assert 'raceLimit' in j.keys()
        assert 'showconf' in j.keys()
        assert 'openlib' in j.keys()
