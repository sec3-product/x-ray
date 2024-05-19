import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) OpenMP races'
if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
    home = '/home/ubuntu/repos'  #use existing repo when testing on aws instance.
else:
    home = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.skip(reason='it took long time,try to optimize it.')
class TestGraphblas:
    proj = "graphblas"
    buildDir_redisGraph = os.path.join(home, 'RedisGraph')
    buildDir_graphblas  = os.path.join(home, 'RedisGraph', 'graphblas_recreate_race')
    logFile_redisGraph  = buildDir_redisGraph + "/at_RedisGraph.log"
    logFile_graphblas   = buildDir_graphblas  + "/at_" + proj + ".log"

    def setup_class(self):
        if 'TEST_ON_AWS_REPORTDIR' not in os.environ.keys():
            self.prepare_RedisGraph()
        self.prepare_Graphblas()

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestGraphblas.logFile_graphblas  + " " + aws_report_dir)
            os.system("cd " + TestGraphblas.buildDir_graphblas + " && \
                      cp .coderrect/logs/log.current " + aws_report_dir + "/log_current_" + TestGraphblas.proj + ".log")

    def prepare_RedisGraph():
        os.system("cd " + home + " && \
                  rm -rf RedisGraph && \
                  git clone --recurse-submodules -j8 https://github.com/RedisGraph/RedisGraph.git && \
                  cd RedisGraph && \
                  coderrect -e redisgraph.so -conf=../redisgraph.json make > " + TestGraphblas.logFile_redisGraph + " 2>&1")

    def prepare_Graphblas():
        os.system("cd " + TestGraphblas.buildDir_redisGraph + " && \
                  rm -rf graphblas_recreate_race && \
                  git clone https://github.com/funemy/graphblas_recreate_race")

    @pytest.mark.timeout(450)
    def test_graphblas(self):
        r = os.system("cd " + TestGraphblas.buildDir_redisGraph + " && \
                      cd graphblas_recreate_race && \
                      coderrect -p " + TestGraphblas.buildDir_redisGraph + " make -j $(nproc) > " + TestGraphblas.logFile_graphblas + " 2>&1")

        assert r == 0

        f = open(TestGraphblas.logFile_graphblas)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
