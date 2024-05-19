import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_Help:
    def test_option_h(self):
        tmpFile = home + '/test_option_h/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_option_h && \
                      mkdir test_option_h && \
                      cd test_option_h && \
                      coderrect -h > " + tmpFile)
        assert r == 0
        f = open(tmpFile)
        t = f.read()
        f.close()

        ret = re.search('-h, -help', t)
        assert ret is not None

        ret = re.search('Display this information.', t)
        assert ret is not None

        ret = re.search('usage:', t)
        assert ret is not None

        ret = re.search('Options:', t)
        assert ret is not None

        ret = re.search('Examples:', t)
        assert ret is not None

    def test_option_help(self):
        tmpFile = home + '/test_option_help/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_option_help && \
                      mkdir test_option_help && \
                      cd test_option_help && \
                      coderrect -help > " + tmpFile)
        assert r == 0
        f = open(tmpFile)
        t = f.read()
        f.close()

        ret = re.search('-h, -help', t)
        assert ret is not None

        ret = re.search('Display this information.', t)
        assert ret is not None

        ret = re.search('usage:', t)
        assert ret is not None

        ret = re.search('Options:', t)
        assert ret is not None

        ret = re.search('Examples:', t)
        assert ret is not None
