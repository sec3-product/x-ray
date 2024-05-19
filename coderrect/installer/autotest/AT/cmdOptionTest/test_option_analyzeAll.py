import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_AnalyzeAll:
    def test_analyzeAll(self):
        tmpFile = home + '/test_analyzeAll/memcached/tmp.log'
        r = os.system("cd " + home + " && \
                      rm -rf test_analyzeAll && \
                      mkdir test_analyzeAll && \
                      cd test_analyzeAll && \
                      git clone https://github.com/memcached/memcached.git && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      coderrect -analyzeAll make -j $(nproc) > " + tmpFile)
        assert r == 0
        with open(tmpFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search('Analyzing \\S+/memcached/memcached-debug', t)
        assert ret is not None

        ret = re.search('Analyzing \\S+/memcached/sizes', t)
        assert ret is not None

        ret = re.search('Analyzing \\S+/memcached/testapp', t)
        assert ret is not None

        ret = re.search('Analyzing \\S+/memcached/timedrun', t)
        assert ret is not None

        ret = re.search('Analyzing \\S+/memcached/memcached', t)
        assert ret is not None
        
        ret = re.search('-+The summary of races in memcached-debug-+', t)
        assert ret is not None
        
        ret = re.search('-+The summary of races in memcached-+', t)
        assert ret is not None

