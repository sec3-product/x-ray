import os
import re
import json

home = os.path.dirname(os.path.realpath(__file__))
coderrect_home = os.getenv('CODERRECT_HOME')
pi_home = coderrect_home + "/examples/pi"

class Test_ReportDir:
    def test_option_o(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_option_o && \
                      mkdir test_option_o && \
                      cd test_option_o && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -o testReport1 clang -fopenmp -g pi.c")
        assert r == 0
        assert os.path.exists(home + '/test_option_o/testReport1/artifacts') is True
        assert os.path.exists(home + '/test_option_o/testReport1/configuration.json') is True
        assert os.path.exists(home + '/test_option_o/testReport1/index.html') is True

    def test_report_outputDir(self):
        r = os.system("cd " + home + " && \
                      rm -rf test_report_outputDir && \
                      mkdir test_report_outputDir && \
                      cd test_report_outputDir && \
                      cp " + pi_home + "/pi.c ./ && \
                      coderrect -report.outputDir=testReport2 clang -fopenmp -g pi.c")
        assert r == 0
        assert os.path.exists(home + '/test_report_outputDir/testReport2/artifacts') is True
        assert os.path.exists(home + '/test_report_outputDir/testReport2/configuration.json') is True
        assert os.path.exists(home + '/test_report_outputDir/testReport2/index.html') is True

