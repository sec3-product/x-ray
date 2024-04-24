#!/usr/bin/env python3
import os.path
from os import path
from re import A, error, sub
import sys
import re
import subprocess

#confusion matrix class
class cm:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    def fdr(self):
       if not (self.fp + self.tp): return 'denominator is zero'
       return str(round(self.fp/(self.fp + self.tp), 2))
    def fpr(self):
        if not (self.fp + self.tn): return 'denominator is zero'
        return str(round(self.fp/(self.fp + self.tn), 2))
   


# category globals
tRaces = tDeadlocks = tMismatchedAPI = ttoctou = tOV = fRaces = fDeadlocks = fMismatchedAPI = ftoctou = fOV = 0
cmRaces = cm()
cmDeadlocks = cm()
cmMismatchedAPI = cm()
cmtoctou = cm()
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

# returns a list of test cases
def getCases():
    cases = []
    i = 0;
    for a in sys.argv:
        if not i: i+=1; continue
        a = findSourceFile(a)
        cases.append(a)
        i += 1
    
    length = len(cases)
    if length:
        return cases
    
    lines = subprocess.getoutput('ls')
    lines = lines.split("\n")
    for line in lines:
        if "apr_"  in line: continue
        if re.findall("(.+)\.cp*$", line):
            cases.append(line)
    return cases

# parse the header of a benchmark case
def parseCase(filepath):
    name = purpose = races = confile = lib = deadlocks = toctou = ov = mismatchedAPI = ''

    findarr = re.findall("(.+)\.cp*$",filepath)
    name = findarr[0]
    tokens = name.split('/')
    name = tokens[len(tokens) - 1]

    fh = open(filepath, 'r')
    for line in fh:
        if len(line) < 5: break
        if re.findall("^//\s+\@purpose\s+(.+)$", line): 
            purpose = re.findall("^//\s+\@purpose\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@dataRaces\s+(.+)$", line): 
            races = re.findall("^//\s+\@dataRaces\s+(.+)$", line)[0]
        
        elif re.findall("^//\s+\@deadlocks\s+(.+)$", line): 
            deadlocks = re.findall("^//\s+\@deadlocks\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@toctou\s+(.+)$", line): 
            toctou = re.findall("^//\s+\@toctou\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@orderViolations\s+(.+)$", line): 
            ov = re.findall("^//\s+\@orderViolations\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@misMatchedAPI\s+(.+)$", line): 
            mismatchedAPI = re.findall("^//\s+\@misMatchedAPI\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@configuration\s+(.+)$", line): 
            confile = re.findall("^//\s+\@configuration\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@lib\s+(.+)$", line): 
            lib = re.findall("^//\s+\@lib\s+(.+)$", line)[0]
    fh.close()
    if not races: races = '0'
    if not deadlocks: deadlocks = '0'
    if not mismatchedAPI: mismatchedAPI = '0'
    if not toctou: toctou = '0'
    if not ov: ov = '0'
    return (name, purpose, races, deadlocks, toctou, ov, mismatchedAPI, confile, lib)

def runRase(filepath, races, deadlocks, toctou, ov, mismatchedAPI, confile, lib):
    global fRaces, fDeadlocks, fMismatchedAPI, ftoctou, fOV, cmRaces, cmDeadlocks, cmMismatchedAPI, cmtoctou, cmOV
    tokens = filepath.split('/')
    executableName = re.findall("(.+)\.cp*$", tokens[len(tokens) - 1])[0]
    
    if confile:
        confile = "../" + confile
    
    subprocess.run(['rm', '-fr', '*'])
    subprocess.run(['rm', '-fr', '.coderrect'])
    rc = ''
    if not confile:
        if not lib:
            rc = subprocess.getstatusoutput("coderrect clang++ -std=gnu++11 " + filepath + " -o " + executableName + " -lpthread")
        else:
            rc = subprocess.getstatusoutput("coderrect ../buildlib " + lib + " " + filepath)
    else:
        if not lib:
            rc = subprocess.getstatusoutput("coderrect -conf=" + confile + " clang++ " + filepath + " -o " + executableName + " -lpthread")
        else:
            rc = subprocess.getstatusoutput("coderrect -conf=" + confile + " ../buildlib " + lib + " " + filepath)

    rc = list(rc)

    rc[1] = rc[1].rstrip()
    if rc[0]:
        return rc[1] + '\n'

    returnstring = ''
    
    # check raw_$executableName.json
    jsonPath = ".coderrect/build/raw_" + executableName + ".json"
    if lib:
        jsonPath = ".coderrect/build/raw_" + lib + ".json"

    # data races check

    if os.path.isfile(jsonPath):
        rc = subprocess.getstatusoutput("jq '.dataRaces | length' " + jsonPath)
        rc = list(rc)
        rc[1] = rc[1].rstrip()
        if rc[0]:
            returnstring += rc[1] + '\n'
            fRaces += int(races)
            cmRaces.fn += int(races)
        elif not rc[1] == races:
            returnstring += 'Found ' + rc[1] + ' data races, expected ' + races + '\n'
            diff = int(races) - int(rc[1])
            fRaces += abs(diff)
            if diff > 0:
                cmRaces.tp += int(rc[1])
                cmRaces.fn += diff
            elif diff < 0:
                cmRaces.tp += int(races)
                cmRaces.fp += abs(diff)
        elif int(rc[1]) == 0 and int(races) == 0:
            cmRaces.tn += 1
        elif rc[1] == races:
            cmRaces.tp += int(rc[1])
    else:
        returnstring += 'Error: Could not find raw json' + '\n'
    
    # deadlocks check
    if os.path.isfile(jsonPath):
        rc = subprocess.getstatusoutput("jq '.deadLocks | length' " + jsonPath)
        rc = list(rc)
        rc[1] = rc[1].rstrip()
        if rc[0]:
            returnstring += rc[1] + '\n'
            fDeadlocks += int(deadlocks)
            cmDeadlocks.fn += int(deadlocks)
        if not rc[1] == deadlocks:
            returnstring += 'Found ' + rc[1] + ' deadlocks, expected ' + deadlocks + '\n'
            diff = int(deadlocks) - int(rc[1])
            fDeadlocks += abs(diff)
            if diff > 0:
                cmDeadlocks.tp += int(rc[1])
                cmDeadlocks.fn += diff
            elif diff < 0:
                cmDeadlocks.tp += int(deadlocks)
                cmDeadlocks.fp += abs(diff)
        elif int(rc[1]) == 0 and int(deadlocks) == 0:
            cmDeadlocks.tn += 1
        elif rc[1] == deadlocks:
            cmDeadlocks.tp += int(rc[1])
    else: 
        returnstring += 'Error: Could not find raw json' + '\n'

    # mismatchedAPI check
    if os.path.isfile(jsonPath):
        rc = subprocess.getstatusoutput("jq '.mismatchedAPIs | length' " + jsonPath)
        rc = list(rc)
        rc[1] = rc[1].rstrip()
        if rc[0]:
            returnstring += rc[1] + '\n'
            fMismatchedAPI += int(mismatchedAPI)
            cmMismatchedAPI.fn += int(mismatchedAPI)
        if not rc[1] == mismatchedAPI:
            returnstring += 'Found ' + rc[1] + ' mismatchedAPIs, expected ' + mismatchedAPI + '\n'
            diff = int(mismatchedAPI) - int(rc[1])
            fMismatchedAPI += abs(diff)
            if diff > 0:
                cmMismatchedAPI.tp += int(rc[1])
                cmMismatchedAPI.fn += diff
            elif diff < 0:
                cmMismatchedAPI.tp += int(mismatchedAPI)
                cmMismatchedAPI.fp += abs(diff)
        elif int(rc[1]) == 0 and int(mismatchedAPI) == 0:
            cmMismatchedAPI.tn += 1
        elif rc[1] == mismatchedAPI:
            cmMismatchedAPI.tp += int(rc[1])
    else: 
         returnstring += 'Error: Could not find raw json' + '\n'

    # orderViolation check
    if os.path.isfile(jsonPath):   
        rc = subprocess.getstatusoutput("jq '.orderViolations | length' " + jsonPath)
        rc = list(rc)
        rc[1] = rc[1].rstrip()
        if rc[0]:
            returnstring += rc[1] + '\n'
            fOV += int(ov)
            cmOV.fn += int(ov)
        if not rc[1] == ov:
            returnstring += 'Found ' + rc[1] + ' OVs, expected ' + ov + '\n'
            diff = int(ov) - int(rc[1])
            fOV += abs(diff)
            if diff > 0:
                cmOV.tp += int(rc[1])
                cmOV.fn += diff
            elif diff < 0:
                cmOV.tp += int(ov)
                cmOV.fp += abs(diff)
        elif int(rc[1]) == 0 and int(ov) == 0:
            cmOV.tn += 1
        elif rc[1] == ov:
            cmOV.tp += int(rc[1])
    else: returnstring += 'Error: Could not find raw json' + '\n'

    # toctou check
    if os.path.isfile(jsonPath):   
        rc = subprocess.getstatusoutput("jq '.toctou | length' " + jsonPath)
        rc = list(rc)
        rc[1] = rc[1].rstrip()
        if rc[0]:
            returnstring += rc[1] + '\n'
            ftoctou += int(toctou)
            cmtoctou.fn += int(toctou)
        if not rc[1] == toctou:
            returnstring += 'Found ' + rc[1] + ' TOCTOUs, expected ' + toctou + '\n'
            diff = int(toctou) - int(rc[1])
            ftoctou += abs(diff)
            if diff > 0:
                cmtoctou.tp += int(rc[1])
                cmtoctou.fn += diff
            elif diff < 0:
                cmtoctou.tp += int(toctou)
                cmtoctou.fp += abs(diff)
        elif int(rc[1]) == 0 and int(toctou) == 0:
            cmtoctou.tn += 1
        elif rc[1] == toctou:
            cmtoctou.tp += int(rc[1])
    else: returnstring += 'Error: Could not find raw json' + '\n'
        

        
    return returnstring.rstrip()

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


for case in cases:
    filepath = '../' + case
    (name, purpose, races, deadlocks, toctou, ov, mismatchedAPI, confile, lib) = parseCase(filepath)
    if (not purpose):
        print('\n' + name + ' - invalid case')
        print('\n----------------------------------------\n')
        continue

    if races: tRaces += int(races)
    if deadlocks: tDeadlocks += int(deadlocks)
    if mismatchedAPI: tMismatchedAPI += int(mismatchedAPI)
    if toctou: ttoctou += int(toctou)
    if ov: tOV += int(ov)
    

    print('Test ' + name + ' - ' + purpose)
    errormsg = runRase(filepath, races, deadlocks, toctou, ov, mismatchedAPI, confile, lib)

    #success/failure conditional

    if errormsg:
        print('\n' + errormsg)
    else:
        print('\n' + 'Success')

    print('\n----------------------------------------\n')

# final report based on category
print("Data race - Failed = " + str(fRaces) + "; Total = " + str(tRaces) + "; FDR = " + cmRaces.fdr() + "; FPR = " + cmRaces.fpr())
print("Deadlock - Failed = " + str(fDeadlocks) + "; Total = " + str(tDeadlocks) + "; FDR = " + cmDeadlocks.fdr() + "; FPR = " + cmDeadlocks.fpr())
print("MismatchedAPI - Failed = " + str(fMismatchedAPI) + "; Total = " + str(tMismatchedAPI) + "; FDR = " + cmMismatchedAPI.fdr() + "; FPR = " + cmMismatchedAPI.fpr())
print("TOCTOUs - Failed = " + str(ftoctou) + "; Total = " + str(ttoctou) + "; FDR = " + cmtoctou.fdr() + "; FPR = " + cmtoctou.fpr())
print("Order Violation - Failed = " + str(fOV) + "; Total = " + str(tOV) + "; FDR = " + cmOV.fdr() + "; FPR = " + cmOV.fpr() + '\n')



# exit code conditional
if fRaces or fDeadlocks or fMismatchedAPI or ftoctou or fOV:
    sys.exit(1)
else:
    sys.exit(0)


# confusion matrix debugging code block

# print("\nData race", cmRaces.tp, cmRaces.tn, cmRaces.fp, cmRaces.fn)
# print("API", cmMismatchedAPI.tp, cmMismatchedAPI.tn, cmMismatchedAPI.fp, cmMismatchedAPI.fn)
# print("cmdeadlock", cmDeadlocks.tp, cmDeadlocks.tn, cmDeadlocks.fp, cmDeadlocks.fn)
# print("toctou", cmtoctou.tp, cmtoctou.tn, cmtoctou.fp, cmtoctou.fn)
# print("OV", cmOV.tp, cmOV.tn, cmOV.fp, cmOV.fn,'\n')