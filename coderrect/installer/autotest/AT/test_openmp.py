import pytest
import os
import re

#result example:
#PASS/FAIL/CRASH
#Total:	 92 / 24 / 0
#using RE to collect final result data
pattern='Total\\s*(\\d+)\\s*(\\d+)'
home = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.skip(reason='Lack of privilege to download LLVMRace.')
class TestOpenmp:
    proj = "openmp"
    buildDir = os.path.join(home, 'LLVMRace/TestCases/dataracebench')
    logFile = buildDir + "/at_" + proj + ".log"
    tmpLog = buildDir + "/tmp.log"

    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf LLVMRace && \
                  git clone git@github.com:coderrect/LLVMRace.git")

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cp " + TestOpenmp.logFile + " " + aws_report_dir)

    @pytest.mark.timeout(240)
    def test_openmp(self):
        r = os.system("cd " + home + " && \
                      cd LLVMRace && \
                      git checkout develop && \
                      cd TestCases/dataracebench && \
                      go run tester.go > " + TestOpenmp.logFile + " 2>&1")

        assert r == 0
        
        os.system("cd " + home + " && \
                  cd LLVMRace/TestCases/dataracebench && \
                  sed -n '/Combined Results/, $p' at_openmp.log > tmp.log")
        f = open(TestOpenmp.tmpLog)
        t = f.read()
        f.close()
        ret = re.search(pattern, t)
        assert ret is not None
        assert int(ret.group(1)) > 100
        #assert int(ret.group(2)) == 0
