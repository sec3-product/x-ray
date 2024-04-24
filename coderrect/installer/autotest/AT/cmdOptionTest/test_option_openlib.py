import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))

class Test_Openlib:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf test_option_openlib && \
                  mkdir test_option_openlib && \
                  cd test_option_openlib && \
                  git clone https://github.com/pjreddie/darknet.git")

    def test_entryPoints(self):
        tmpFile = home + '/test_option_openlib/darknet/tmp.log'
        r = os.system("cd " + home + " && \
                      cd test_option_openlib/darknet && \
                      coderrect -e libdarknet.a -conf=../../openlib.json make -j $(nproc) > " + tmpFile)
        assert r == 0
        f = open(tmpFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()

        ret = re.search('EntryPoints:\nalphanum_to_int\ngemm_bin\nresize_data\n', t)
        assert ret is not None

        ret = re.search('-+The summary of races in libdarknet.a-+', t)
        assert ret is not None

        ret = re.search('(\\d+) shared data races', t)
        assert ret is not None
        assert int(ret.group(1)) > 0
