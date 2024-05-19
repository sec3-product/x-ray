import pytest
import os
import re

expected_no_race = 'No race detected'
expected_race = '\\s+(\\d+) shared data races'
expected_mismatch = '\\s+(\\d+) mis-matched API issues'

home = os.path.dirname(os.path.realpath(__file__))

class Test_TDAPIProf:

    def setup_class(self):
        r = os.system("cd " + home + " && \
                      gcc -shared -fPIC micthread.c -o libmicthread.so -lpthread")
        assert r == 0

    def teardown_class(self):
        if 'TEST_ON_AWS_REPORTDIR' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cd " + home + " && \
                      cp at_*.log " + aws_report_dir)

    @pytest.mark.timeout(30)
    def test_mutex_norace(self):
        logFile = home + '/at_TAP_mutex_no_race.log'
        r = os.system("cd " + home + " && \
                      rm -rf TAP_mutex_no_race && \
                      mkdir TAP_mutex_no_race && \
                      cd TAP_mutex_no_race && \
                      coderrect -conf=../micthread.json gcc -o TAP_mutex_no_race ../TAP_mutex_no_race.c -L../ -lmicthread > " + logFile)

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_mutex_race(self):
        logFile = home + '/at_TAP_mutex_race.log'
        r = os.system("cd " + home + " && \
                      rm -rf TAP_mutex_race && \
                      mkdir TAP_mutex_race && \
                      cd TAP_mutex_race && \
                      coderrect -conf=../micthread.json gcc -o TAP_mutex_race ../TAP_mutex_race.c -L../ -lmicthread > " + logFile)

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) == 1

    @pytest.mark.timeout(30)
    def test_rwlock_no_race(self):
        logFile = home + '/at_TAP_rwlock_no_race.log'
        r = os.system("cd " + home + " && \
                      rm -rf TAP_rwlock_no_race && \
                      mkdir TAP_rwlock_no_race && \
                      cd TAP_rwlock_no_race && \
                      coderrect -conf=../micthread.json gcc -o TAP_rwlock_no_race ../TAP_rwlock_no_race.c -L../ -lmicthread > " + logFile)

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_rwlock_race(self):
        logFile = home + '/at_TAP_rwlock_race.log'
        r = os.system("cd " + home + " && \
                      rm -rf TAP_rwlock_race && \
                      mkdir TAP_rwlock_race && \
                      cd TAP_rwlock_race && \
                      coderrect -conf=../micthread.json gcc -o TAP_rwlock_race ../TAP_rwlock_race.c -L../ -lmicthread > " + logFile)

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) == 1

    @pytest.mark.timeout(30)
    def test_spinlock_no_race(self):
        logFile = home + '/at_TAP_spinlock_no_race.log'
        r = os.system("cd " + home + " && \
                      rm -rf TAP_spinlock_no_race && \
                      mkdir TAP_spinlock_no_race && \
                      cd TAP_spinlock_no_race && \
                      coderrect -conf=../micthread.json g++ -o TAP_spinlock_no_race ../TAP_spinlock_no_race.cpp -L../ -lmicthread > " + logFile)

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_spinlock_race(self):
        logFile = home + '/at_TAP_spinlock_race.log'
        r = os.system("cd " + home + " && \
                      rm -rf TAP_spinlock_race && \
                      mkdir TAP_spinlock_race && \
                      cd TAP_spinlock_race && \
                      coderrect -conf=../micthread.json g++ -o TAP_spinlock_race ../TAP_spinlock_race.cpp -L../ -lmicthread > " + logFile)

        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        ret = re.search(expected_mismatch, t)
        assert ret is not None
        assert int(ret.group(1)) == 1
