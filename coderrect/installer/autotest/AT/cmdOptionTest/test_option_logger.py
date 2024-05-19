import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))
coderrect_home = os.getenv('CODERRECT_HOME')
pi_home = coderrect_home + "/examples/pi"

class Test_Logger:
    def test_loggerLevel(self):
        log_current = home + "/test_loggerLevel/.coderrect/logs/log.current"
        r = os.system("cd " + home + " && \
                      rm -rf test_loggerLevel && \
                      mkdir test_loggerLevel && \
                      cd test_loggerLevel && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -logger.level=info clang -fopenmp -g pi.c")
        assert r == 0
        f = open(log_current, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('\\[debug\\]', t)
        assert ret is None

        ret = re.search('\\[info\\]', t)
        assert ret is not None

    def test_logFolder(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_logFolder && \
                      mkdir test_logFolder && \
                      cd test_logFolder && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -logger.logFolder=abc/logs clang -fopenmp -g pi.c")
        assert r == 0
        assert os.path.exists(home + '/test_logFolder/.coderrect/abc/logs/log.current') is True

    def test_toStderr(self):
        tmpFile = home + '/test_toStderr/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_toStderr && \
                      mkdir test_toStderr && \
                      cd test_toStderr && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -logger.toStderr=true clang -fopenmp -g pi.c 2>" + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('\\[debug\\]', t)
        assert ret is not None

        ret = re.search('\\[info\\]', t)
        assert ret is not None
