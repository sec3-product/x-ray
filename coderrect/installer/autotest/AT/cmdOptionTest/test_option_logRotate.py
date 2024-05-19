import os
import re
import json
import pytest

home = os.path.dirname(os.path.realpath(__file__))

class Test_LogRotate:
    def setup_class(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_LogRotate && \
                      mkdir test_LogRotate && \
                      cd test_LogRotate && \
                      git clone https://github.com/memcached/memcached.git && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      coderrect -e memcached,memcached-debug,testapp -conf=../../logRotate.json make -j $(nproc)")
        assert r == 0

    def teardown_class(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_LogRotate/")
        assert r == 0

    @pytest.mark.timeout(200)
    def test_log_rotate(self):
        
        #rotate 1st time
        r = os.system("cd " + home + " && \
                      cd test_LogRotate/memcached && \
                      make clean && \
                      coderrect -e memcached,memcached-debug,testapp -conf=../../logRotate.json make -j $(nproc)")
        
        assert os.path.exists(home + '/test_LogRotate/memcached/.coderrect/logs/log.0') is True

        #rotate 2nd time
        r = os.system("cd " + home + " && \
                      cd test_LogRotate/memcached && \
                      make clean && \
                      coderrect -e memcached,memcached-debug,testapp -conf=../../logRotate.json make -j $(nproc)")

        assert os.path.exists(home + '/test_LogRotate/memcached/.coderrect/logs/log.1') is True

        #rotate 3rd time
        r = os.system("cd " + home + " && \
                      cd test_LogRotate/memcached && \
                      make clean && \
                      coderrect -e memcached,memcached-debug,testapp -conf=../../logRotate.json make -j $(nproc)")

        assert os.path.exists(home + '/test_LogRotate/memcached/.coderrect/logs/log.2') is not True 
