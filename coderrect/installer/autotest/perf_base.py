import pytest
import time
import os
import re

expected_no_race = 'No race is detected.'
expected_race = 'detected (\\d+) races in total.'

class TestCoderrect():
    home = os.getcwd()
    prefix = 'coderrect-linux'
    pkg = ''
    report = ''

    start = 0
    end = 0

    def setup_class(self):
        files = os.listdir()
        for f in files:
            if TestCoderrect.prefix in f and os.path.isdir(f):
                TestCoderrect.pkg = TestCoderrect.home + "/" + f + "/examples/"
                TestCoderrect.report = TestCoderrect.home + "/report"
                break
        if TestCoderrect.pkg == '':
            print("Missing coderrect package, pls check it.")
            exit(1)
        os.system("cd " + TestCoderrect.home + " && \
                  rm -rf report && \
                  rm -f report.tar.gz && \
                  mkdir report")

    def teardown_class(self):
        print("\n\nPackage logs:")
        os.system("cd " + TestCoderrect.home + " && \
                  tar -czvf report.tar.gz report")

    def setup_method(self):
        self.start = int(time.time())

    def teardown_method(self):
        self.end = int(time.time())
        print("duration time: " + str(self.end - self.start))

    # def test_hello_c(self):
    #     print("\ntest case: test_hello_c", end=" ")
    #     path = TestCoderrect.pkg + "hello"
    #     logFile = path + "/at_hello.log"
    #     r = os.system("cd " + path + " && \
    #                   coderrect clang -fopenmp -g hello.c > " + logFile + " 2>&1")
    #     os.system("cp " + logFile + " " + TestCoderrect.report)
    #     os.system("cd " + path + " && \
    #               cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_helloc.log")
    #     assert r == 0
    #
    #     f = open(logFile)
    #     t = f.read()
    #     f.close()
    #     assert expected_no_race in t
    #
    # def test_pi_c(self):
    #     print("\ntest case: test_pi_c", end=" ")
    #     path = TestCoderrect.pkg + "pi"
    #     logFile = path + "/at_pi.log"
    #     r = os.system("cd " + path + " && \
    #                   coderrect clang -fopenmp -g pi.c > " + logFile + " 2>&1")
    #     os.system("cp " + logFile + " " + TestCoderrect.report)
    #     os.system("cd " + path + " && \
    #               cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_pic.log")
    #     assert r == 0
    #
    #     f = open(logFile)
    #     t = f.read()
    #     f.close()
    #     ret = re.search(expected_race, t)
    #     assert ret is not None
    #     assert int(ret.group(1)) > 0
    #
    # def test_singlemake(self):
    #     print("\ntest case: test_singlemake", end=" ")
    #     path = TestCoderrect.pkg + "singlemake"
    #     logFile = path + "/at_singlemake.log"
    #     r = os.system("cd " + path + " && \
    #                   coderrect . > " + logFile + " 2>&1")
    #     os.system("cp " + logFile + " " + TestCoderrect.report)
    #     os.system("cd " + path + " && \
    #               cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_singlemake.log")
    #     assert r == 0
    #
    #     f = open(logFile)
    #     t = f.read()
    #     f.close()
    #     assert expected_no_race in t

    @pytest.mark.repeat(10)
    def test_cblosc(self):
        print("\ntest case: test_cblosc", end=" ")
        proj = "c-blosc"
        path = TestCoderrect.home + "/" + proj
        logFile = path + "/build/at_" + proj + ".log"
        r = os.system("cd " + path + " && \
                      git checkout v1.17.1 && \
                      rm -rf build && \
                      mkdir build && \
                      cd build && \
                      cmake .. > " + logFile + " 2>&1 && \
                      coderrect make -j $(nproc) >> " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp build/.coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.repeat(10)
    def test_curl(self):
        print("\ntest case: test_curl", end=" ")
        proj = "curl"
        path = TestCoderrect.home + "/" + proj
        logFile = path + "/build/at_" + proj + ".log"
        r = os.system("cd " + path + " && \
                      git checkout curl-7_69_0 && \
                      rm -rf build && \
                      mkdir build && \
                      cd build && \
                      cmake .. > " + logFile + " 2>&1 && \
                      coderrect make -j $(nproc) >> " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp build/.coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.repeat(10)
    def test_DGtal(self):
        print("\ntest case: test_DGtal", end=" ")
        proj = "DGtal"
        path = TestCoderrect.home + "/" + proj
        logFile = path + "/build/at_" + proj + ".log"
        r = os.system("cd " + path + " && \
                      git checkout v0.9 && \
                      rm -rf build && \
                      mkdir build && \
                      cd build && \
                      cmake -DWITH_OPENMP=ON -DBUILD_TESTING=ON .. > " + logFile + " 2>&1 && \
                      coderrect make -j $(nproc) >> " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp build/.coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        assert expected_no_race in t

    @pytest.mark.repeat(10)
    def test_Darknet(self):
        print("\ntest case: test_Darknet", end=" ")
        proj = "darknet"
        path = TestCoderrect.home + "/" + proj
        logFile = path + "/at_" + proj + ".log"
        r = os.system("cd " + TestCoderrect.home + " && \
                      rm -rf darknet && \
                      tar -xf darknet.tar && \
                      cd " + path + " && \
                      coderrect make -j $(nproc) > " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.repeat(10)
    def test_graphblas(self):
        print("\ntest case: test_graphblas", end=" ")
        proj_grr = "graphblas_recreate_race"
        path_grr = TestCoderrect.home + "/RedisGraph/" + proj_grr
        logFile_grr = path_grr + "/at_" + proj_grr + ".log"
        r = os.system("cd " + TestCoderrect.home + " && \
                      rm -rf RedisGraph && \
                      tar -xf RedisGraph.tar && \
                      cd " + path_grr + " && \
                      coderrect make -j $(nproc) > " + logFile_grr + " 2>&1")
        os.system("cp " + logFile_grr + " " + TestCoderrect.report)
        os.system("cd " + path_grr + " && \
                  cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj_grr + ".log")
        assert r == 0

        f = open(logFile_grr)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.repeat(10)
    def test_pbzip2(self):
        print("\ntest case: test_pbzip2", end=" ")
        proj = "pbzip2"
        path = TestCoderrect.home + "/sctbench/benchmarks/conc-bugs/pbzip2-0.9.4"
        logFile = path + "/pbzip2-0.9.4/at_" + proj + ".log"
        r = os.system("cd " + path + " && \
                      rm -rf pbzip2-0.9.4 && \
                      tar -xf pbzip2-0.9.4.tar && \
                      cd pbzip2-0.9.4 && \
                      coderrect make -j $(nproc) > " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp pbzip2-0.9.4/.coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.repeat(10)
    def test_lulesh(self):
        print("\ntest case: test_lulesh", end=" ")
        proj = "lulesh"
        path = TestCoderrect.home + "/LULESH-2.0.3"
        logFile = path + "/at_" + proj + ".log"
        r = os.system("cd " + TestCoderrect.home + " && \
                      rm -rf LULESH-2.0.3 && \
                      tar -xf LULESH-2.0.3.tar && \
                      cd " + path + " && \
                      coderrect make -j $(nproc) > " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.repeat(10)
    def test_Quicksilver(self):
        print("\ntest case: test_Quicksilver", end=" ")
        proj = "Quicksilver"
        path = TestCoderrect.home + "/Quicksilver/src"
        logFile = path + "/at_" + proj + ".log"
        r = os.system("cd " + TestCoderrect.home + " && \
                      rm -rf Quicksilver && \
                      tar -xf Quicksilver.tar && \
                      cd " + path + " && \
                      coderrect make -j $(nproc) > " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0

    @pytest.mark.repeat(10)
    def test_miniAMR(self):
        print("\ntest case: test_miniAMR", end=" ")
        proj = "miniAMR"
        path = TestCoderrect.home + "/miniAMR-1.5.0/openmp"
        logFile = path + "/at_" + proj + ".log"
        r = os.system("cd " + TestCoderrect.home + " && \
                      rm -rf miniAMR-1.5.0 && \
                      tar -xf miniAMR-1.5.0.tar && \
                      cd " + path + " && \
                      coderrect make -j $(nproc) > " + logFile + " 2>&1")
        os.system("cp " + logFile + " " + TestCoderrect.report)
        os.system("cd " + path + " && \
                  cp .coderrect/logs/log.current " + TestCoderrect.report + "/log_current_" + proj + ".log")
        assert r == 0

        f = open(logFile)
        t = f.read()
        f.close()
        ret = re.search(expected_race, t)
        assert ret is not None
        assert int(ret.group(1)) > 0
