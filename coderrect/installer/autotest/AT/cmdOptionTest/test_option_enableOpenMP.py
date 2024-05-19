import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))
coderrect_home = os.getenv('CODERRECT_HOME')

class Test_EnableOpenMP:
    def test_enableOpenMP(self):
        tmpFile = home + '/test_enableOpenMP/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_enableOpenMP && \
                    mkdir test_enableOpenMP && \
                    cd test_enableOpenMP && \
                    cp " + coderrect_home + "/examples/pi/pi.c ./ && \
                    coderrect -enableOpenMP=false clang -fopenmp -g pi.c > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('No race detected', t)
        assert ret is not None

