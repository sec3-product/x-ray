import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = 'detected (\\d+) races in total.'
home = os.getcwd()
compat_dir = home + os.sep + 'compat_log'
report_dir = home + os.sep + 'report'
images=["centostest:7.0.1406", "ubuntutest:1604"]

@pytest.mark.parametrize('i', images)
class TestCompatible():
    def setup_class(self):
        os.system("rm -rf compat_log* && \
                  mkdir compat_log")

    def teardown_class(self):
        os.system("tar -czvf compat_log.tar.gz compat_log")

    def setup_method(self):
        os.system("rm -rf darknet && \
                  tar -xf darknet.tar")
    
    @pytest.mark.timeout(48)
    def test_compatible(self, i):
        print("\nTest on docker image: " + i)
        file = i + "_darknet.log"
        logInDocker = "/compat_test/compat_log/" + file
        logOutOfDocker = compat_dir + os.sep + file
        r = os.system("docker run \
                  --rm \
                  --user=$(id -u):$(id -g) \
                  -v /home/ubuntu/repos:/compat_test \
                  " + i + " \
                  sh -c 'cd /compat_test/darknet && coderrect make -j $(nproc) > " + logInDocker + " 2>&1'")
        os.system("cd darknet && \
                   cp .coderrect/logs/log.current " + compat_dir + "/" + i + "_log_current_darknet.log")
        assert r == 0

        f = open(logOutOfDocker)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
