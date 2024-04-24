import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_TerminalReport:
    def test_option_t(self):
        tmpFile = home + '/test_option_t/darknet/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_option_t && \
                    mkdir test_option_t && \
                    cd test_option_t && \
                    git clone https://github.com/pjreddie/darknet.git && \
                    cd darknet && \
                    coderrect -t -e darknet make -j $(nproc) > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('==== Found a race between:', t)
        assert ret is not None

        ret = re.search('Shared variable:', t)
        assert ret is not None

        ret = re.search('Thread 1:', t)
        assert ret is not None
        
        ret = re.search('Thread 2:', t)
        assert ret is not None

        ret = re.search('>>>Stack Trace:', t)
        assert ret is not None

    def test_enableTerminal(self):
        tmpFile = home + '/test_enableTerminal/darknet/tmp.log'
        r = os.system("cd " + home + " && \
                    rm -rf test_enableTerminal && \
                    mkdir test_enableTerminal && \
                    cd test_enableTerminal && \
                    git clone https://github.com/pjreddie/darknet.git && \
                    cd darknet && \
                    coderrect -report.enableTerminal -e darknet make -j $(nproc) > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('==== Found a race between:', t)
        assert ret is not None

        ret = re.search('Shared variable:', t)
        assert ret is not None

        ret = re.search('Thread 1:', t)
        assert ret is not None

        ret = re.search('Thread 2:', t)
        assert ret is not None

        ret = re.search('>>>Stack Trace:', t)
        assert ret is not None
