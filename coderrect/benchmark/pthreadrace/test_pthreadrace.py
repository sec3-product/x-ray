import pytest
import os
import re

expected_no_race = 'No race is detected'
expected_race = '\\s+(\\d+) shared data races'

class NonRun_TestPthread():
    home = os.path.dirname(os.path.realpath(__file__))
    reportDir = home + "/build/at_report"
    buildDir = home + "/build"

    def setup_class(self):
        print("\n")
        os.system("cd " + TestPthread.home + " && \
                  rm -rf build && \
                  mkdir build && \
                  cd build && \
                  mkdir at_report && \
                  cmake ..")

    def teardown_class(self):
        if 'TEST_ON_AWS' in os.environ.keys():
            aws_report_dir = os.environ['TEST_ON_AWS_REPORTDIR']
            os.system("cd " + TestPthread.reportDir + " && \
                      cp *.log " + aws_report_dir)

    def setup_method(self):
        os.system("cd " + TestPthread.buildDir + " && \
                  make clean && \
                  rm -f .coderrect/logs/log.current")

    @pytest.mark.timeout(30)
    def test_pthreadrace(self):
        executable = 'pthreadrace'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                      coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                  cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_correct(self):
        executable = 'pthreadrace_correct'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_correct.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_deadlock(self):
        executable = 'pthreadrace_deadlock'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_deadlock.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_fine(self):
        executable = 'pthreadrace_fine'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_fine.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_lock_guard(self):
        executable = 'pthreadrace_lock_guard'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_lock_guard.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.timeout(30)
    def test_pthreadrace_miss_signal(self):
        executable = 'pthreadrace_miss_signal'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_miss_signal.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_rwlock(self):
        executable = 'pthreadrace_rwlock'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_rwlock.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_rwlock_int(self):
        executable = 'pthreadrace_rwlock_int'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_rwlock_int.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_safe(self):
        executable = 'pthreadrace_safe'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_safe.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_pthreadrace_sem(self):
        executable = 'pthreadrace_sem'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_pthreadrace_sem.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.timeout(30)
    def test_stdthreadrace(self):
        executable = 'stdthreadrace'
        print("\ntest case: test_" + executable, end=" ")
        logFile = TestPthread.reportDir + "/at_stdthreadrace.log"
        r = os.system("cd " + TestPthread.buildDir + " && \
                          coderrect -e " + executable + " make > " + logFile + " 2>&1")
        os.system("cd " + TestPthread.buildDir + " && \
                      cp .coderrect/logs/log.current " + TestPthread.reportDir + "/log_current_" + executable + ".log")
        assert r == 0

        with open(logFile, 'r', encoding='utf-8') as f:
            t = f.read()

        assert expected_no_race in t
