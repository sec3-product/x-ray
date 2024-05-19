import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_publishResults:
    def test_publishResults_0(self):
        tmpFile = home + '/test_publishResults_0/memcached/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_publishResults_0 && \
                    mkdir test_publishResults_0 && \
                    cd test_publishResults_0 && \
                    git clone https://github.com/memcached/memcached.git && \
                    cd memcached && \
                    ./autogen.sh && \
                    ./configure && \
                    make clean && \
                    coderrect -e testapp make -j $(nproc) && \
                    coderrect -publish.jenkins; echo $? > " + tmpFile)
        assert r == 0
        f = open(tmpFile)
        t = f.read()
        f.close()

        ret = re.search('0', t)
        assert ret is not None

    def test_publishResults_1(self):
        tmpFile = home + '/test_publishResults_1/memcached/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_publishResults_1 && \
                    mkdir test_publishResults_1 && \
                    cd test_publishResults_1 && \
                    git clone https://github.com/memcached/memcached.git && \
                    cd memcached && \
                    ./autogen.sh && \
                    ./configure && \
                    make clean && \
                    coderrect -e memcached make -j $(nproc) && \
                    coderrect -publish.jenkins; echo $? > " + tmpFile)
        assert r == 0

        f = open(tmpFile)
        t = f.read()
        f.close()

        ret = re.search('1', t)
        assert ret is not None
