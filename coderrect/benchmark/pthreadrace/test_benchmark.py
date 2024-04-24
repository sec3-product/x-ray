import pytest
import os
import re

expected='Total: \\d+; Succeed: \\d+; Failed: (\\d+)'

class TestBenchmark():
    home = os.path.dirname(os.path.realpath(__file__))
    logFile = home + "/at_benchmark.log"

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cd " + TestBenchmark.home + " && \
                      cp " + TestBenchmark.logFile + " " + aws_report_dir)

    @pytest.mark.timeout(240)
    def test_benchmark(self):
        executable = 'benchmark'
        print("\ntest case: test_" + executable, end=" ")
        r = os.system("cd " + TestBenchmark.home + " && \
                      ./benchmark > " + TestBenchmark.logFile + " 2>&1")
        #assert r == 0
    
        f = open(TestBenchmark.logFile, 'r', encoding='utf-8')
        t = f.read()
        f.close()
        ret = re.search(expected, t)
        assert ret is not None
        assert int(ret.group(1)) <= 16
