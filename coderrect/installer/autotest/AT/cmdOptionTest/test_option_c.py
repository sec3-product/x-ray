import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_CleanBuild:
    def test_option_c(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_option_c && \
                      mkdir test_option_c && \
                      cd test_option_c && \
                      git clone https://github.com/memcached/memcached.git && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      coderrect -e memcached make -j $(nproc)")
        assert r == 0
        
        r = os.popen("cd " + home + " && \
                     cd test_option_c && \
                     ls -la memcached/.coderrect/build/ | grep '.o.bc$'")
        assert len(r.readlines()) > 0
        
        os.system("cd " + home + " && \
                  cd test_option_c/memcached && \
                  coderrect -c")

        r = os.popen("cd " + home + " && \
                     cd test_option_c && \
                     ls -la memcached/.coderrect/build/ | grep '.o.bc$'")
        assert len(r.readlines()) == 0

    def test_cleanBuild(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_cleanBuild && \
                      mkdir test_cleanBuild && \
                      cd test_cleanBuild && \
                      git clone https://github.com/memcached/memcached.git && \
                      cd memcached && \
                      ./autogen.sh && \
                      ./configure && \
                      coderrect -e memcached make -j $(nproc)")
        assert r == 0

        r = os.popen("cd " + home + " && \
                     cd test_cleanBuild && \
                     ls -la memcached/.coderrect/build/ | grep '.o.bc$'")
        assert len(r.readlines()) > 0
        
        os.system("cd " + home + " && \
                  cd test_cleanBuild/memcached && \
                  coderrect -cleanBuild")

        r = os.popen("cd " + home + " && \
                     cd test_cleanBuild && \
                     ls -la memcached/.coderrect/build/ | grep '.o.bc$'")
        assert len(r.readlines()) == 0
