import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_Executable:
    def test_option_e(self):
        tmpFile = home + '/test_option_e/redis/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_option_e && \
                    mkdir test_option_e && \
                    cd test_option_e && \
                    git clone https://github.com/redis/redis.git && \
                    cd redis && \
                    git checkout 6.0 && \
                    make distclean && \
                    coderrect -e redis-server,redis-cli make MALLOC=libc > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('Analyzing \\S+/redis/src/redis-server', t)
        assert ret is not None

        ret = re.search('Analyzing \\S+/redis/src/redis-cli', t)
        assert ret is not None

        ret = re.search('-+The summary of races in redis-server-+', t)
        assert ret is not None

        ret = re.search('(\\d+) shared data races', t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    def test_analyzeBinaries(self):
        tmpFile = home + '/test_analyzeBinaries/redis/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_analyzeBinaries && \
                    mkdir test_analyzeBinaries && \
                    cd test_analyzeBinaries && \
                    git clone https://github.com/redis/redis.git && \
                    cd redis && \
                    git checkout 6.0 && \
                    make distclean && \
                    coderrect -analyzeBinaries=redis-server,redis-cli make MALLOC=libc > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('Analyzing \\S+/redis/src/redis-server', t)
        assert ret is not None

        ret = re.search('Analyzing \\S+/redis/src/redis-cli', t)
        assert ret is not None

        ret = re.search('-+The summary of races in redis-server-+', t)
        assert ret is not None

        ret = re.search('(\\d+) shared data races', t)
        assert ret is not None
        assert int(ret.group(1)) > 0
