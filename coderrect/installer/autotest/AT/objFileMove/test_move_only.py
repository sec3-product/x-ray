import pytest
import os
import re

expected_no_race = 'No race is detected.'
expected_race = '\\s+(\\d+) shared data races'

home = os.path.dirname(os.path.realpath(__file__))
proj = 'test_move_only'
buildDir = os.path.join(home, proj, 'c-move')
logFile = buildDir + '/test_move_only.log'
tmpFile = buildDir + '/tmp.log'

class TestMove:
    def setup_class(self):
        os.system("cd " + home + " && \
                  rm -rf test_move_only && \
                  mkdir test_move_only && \
                  cp -r c-move test_move_only/")

    def teardown_class(self):
        pass

    def prepare_env(self, subdirs, targets):
        r = os.system("cd " + home + " && \
                  cd test_move_only/c-move && \
                  cp Makefile.template.move Makefile && \
                  sed -i 's/DIR_HOLD_PLACE/" + subdirs + "/' Makefile && \
                  sed -i 's/TARGET_HOLD_PLACE/" + targets + "/' Makefile")
        assert r == 0

    def run_case(self):
        executable_list = [['libfoo.a', 'expected_race', 1],
                           ['main_mvs', 'no_race',       0],
                           ['main',     'expected_race', 1],
                           ['libmvs.a', 'no_race',       0],
                           ['libbar.a', 'no_race',       0]]

        r = os.system("cd " + home + " && \
                      cd test_move_only/c-move && \
                      make clean && \
                      rm -f " + logFile + " && \
                      coderrect -e libfoo.a,main_mvs,main,libmvs.a,libbar.a -conf=./c-move.json make > " + logFile + " 2>&1")

        assert r == 0
   
        for executable in executable_list: 
            t = os.system("cd " + buildDir + " && \
                          rm -f tmp.log && \
                          grep -A 3 'The summary of races in " + executable[0] + "' " + logFile + " > tmp.log")

            with open(tmpFile, 'r', encoding='utf-8') as f:
                t = f.read()
            
            if executable[1] == 'expected_race':
                ret = re.search(expected_race, t)
                assert ret is not None
                assert int(ret.group(1)) == executable[2]
            else:
                ret = re.search(expected_no_race, t)
                assert ret is not None

    @pytest.mark.timeout(240)
    def test_move(self):
        case_list = [['bar foo', 'main test_mvs'],
                     ['bar foo', 'test_mvs main'],
                     ['foo bar', 'main test_mvs'],
                     ['foo bar', 'test_mvs main']]
                                       
        for case in case_list:
            self.prepare_env(case[0], case[1])
            self.run_case()
            print("case: dirs=" + case[0] + ", targets=" + case[1] + "is successful.")


