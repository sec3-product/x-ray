#!/usr/bin/env python3
import os.path
from os import path
from re import A, error, sub
import sys
import re
import subprocess

class cm:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    def fpr(self):
        if not (self.fp + self.tn): return 'No FPR - denominator is zero'
        return str(round(self.fp/(self.fp + self.tn), 2))
    def fdr(self):
        if not (self.fp + self.tp): return 'No FDR - denominator is zero'
        return str(round(self.fp/(self.fp + self.tp), 2))

# category globals
tRaces = tDeadlocks = tMismatchedAPI = tAV = tOV = fRaces = fDeadlocks = fMismatchedAPI = fAV = fOV = 0
cmRaces = cm()
cmDeadlocks = cm()
cmMismatchedAPI = cm()
cmAV = cm()
cmOV = cm()

# find the corresponding source file of the given
# benchmark case. It terminates the program
# when fails
def findSourceFile(case):
    if ".c" in case or ".cpp" in case:
        assert path.exists(case), "Unable to find the file " + case
        return case

    if path.exists(case + ".c"):
        return case + ".c"

    if path.exists(case + ".cpp"):
        return case + ".cpp"
    
    assert "Unable to find the file for case " + case



# main()

# get a list of cases
cases = getCases()

# create the working directory
if not path.exists('build'):
    current = os.getcwd()
    build = 'build'
    try:
        os.mkdir(path.join(current, build))
    except IOError as e:
        print("Failed to create 'build' directory - " + e)

try:
    os.chdir('build')
except IOError as e:
    print("Failed to change the working directory - " + e)