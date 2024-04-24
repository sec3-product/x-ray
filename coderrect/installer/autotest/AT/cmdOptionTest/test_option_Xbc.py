import os
import re
import json
import pytest

home = os.path.dirname(os.path.realpath(__file__))

class Test_B_Xbc:
    def test_option_XbcOnly_Xbc(self):
        #check XbcOnly
        r = os.system("cd " + home + " && \
                    rm -rf test_option_XbcOnly_Xbc && \
                    mkdir test_option_XbcOnly_Xbc && \
                    cd test_option_XbcOnly_Xbc && \
                    git clone https://github.com/pjreddie/darknet.git && \
                    cd darknet && \
                    coderrect -XbcOnly make -j $(nproc)")
        assert r == 0

        assert os.path.exists(home + '/test_option_XbcOnly_Xbc/darknet/.coderrect/build/darknet.bc') is True
        
        #check Xbc
        tmpFile = home + '/test_option_XbcOnly_Xbc/darknet/tmp.log'
        r = os.system("coderrect -Xbc=" + home + "/test_option_XbcOnly_Xbc/darknet/.coderrect/build/darknet.bc > " + tmpFile) 
        assert r == 0

        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('The summary of races in darknet.bc', t)
        assert ret is not None

        ret = re.search('\\s+(\\d+) shared data races', t)
        assert ret is not None
        assert int(ret.group(1)) > 0
