import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_Version:
    def getSearchFmt(self):
        coderrect_home = os.getenv("CODERRECT_HOME")
        pkg = coderrect_home.split('/')[-1]
        pkg_fields = pkg.split('-')
        if len(pkg_fields) > 3:
            fmt = pkg_fields[-2] + '.*' + pkg_fields[-1]
        else:
            fmt = pkg_fields[-1]
        return fmt

    def test_option_v(self):
        tmpFile = home + '/test_option_v/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_option_v && \
                      mkdir test_option_v && \
                      cd test_option_v && \
                      coderrect -v > " + tmpFile)
        assert r == 0
        f = open(tmpFile)
        t = f.read()
        f.close()
       
        regx = self.getSearchFmt()
        ret = re.search(regx, t)
        assert ret is not None

    def test_option_version(self):
        tmpFile = home + '/test_option_version/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_option_version && \
                      mkdir test_option_version && \
                      cd test_option_version && \
                      coderrect -version > " + tmpFile)
        assert r == 0
        f = open(tmpFile)
        t = f.read()
        f.close()

        regx = self.getSearchFmt()
        ret = re.search(regx, t)
        assert ret is not None
