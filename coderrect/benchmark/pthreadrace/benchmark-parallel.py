#!/usr/bin/env python3
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import os.path
from os import path
from re import A, error, sub
import sys
import re
import subprocess
import pathlib

# confusion matrix class


class cm:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def fdr(self):
        if not (self.fp + self.tp):
            return 'denominator is zero'
        return str(round(self.fp/(self.fp + self.tp), 2))

    def fpr(self):
        if not (self.fp + self.tn):
            return 'denominator is zero'
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
    i = 0
    for a in sys.argv:
        if not i:
            i += 1
            continue
        a = findSourceFile(a)
        cases.append(a)
        i += 1

    length = len(cases)
    if length:
        return cases

    lines = subprocess.getoutput('ls')
    lines = lines.split("\n")
    for line in lines:
        if "apr_" in line:
            continue
        if re.findall("(.+)\.cp*$", line):
            cases.append(line)
    return cases

# parse the header of a benchmark case


def parseCase(filepath):
    name = purpose = races = confile = lib = deadlocks = toctou = ov = mismatchedAPI = ''

    findarr = re.findall("(.+)\.cp*$", filepath)
    name = findarr[0]
    tokens = name.split('/')
    name = tokens[len(tokens) - 1]

    fh = open(filepath, 'r')
    for line in fh:
        if len(line) < 5:
            break
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
            mismatchedAPI = re.findall(
                "^//\s+\@misMatchedAPI\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@configuration\s+(.+)$", line):
            confile = re.findall("^//\s+\@configuration\s+(.+)$", line)[0]

        elif re.findall("^//\s+\@lib\s+(.+)$", line):
            lib = re.findall("^//\s+\@lib\s+(.+)$", line)[0]
    fh.close()
    if not races:
        races = '0'
    if not deadlocks:
        deadlocks = '0'
    if not mismatchedAPI:
        mismatchedAPI = '0'
    if not toctou:
        toctou = '0'
    if not ov:
        ov = '0'
    return (name, purpose, races, deadlocks, toctou, ov, mismatchedAPI, confile, lib)


def runRase(filepath, baseDir, workDir, races, deadlocks, toctou, ov, mismatchedAPI, confile, lib):
    global fRaces, fDeadlocks, fMismatchedAPI, ftoctou, fOV, cmRaces, cmDeadlocks, cmMismatchedAPI, cmtoctou, cmOV
    tokens = filepath.split('/')
    executableName = re.findall("(.+)\.cp*$", tokens[len(tokens) - 1])[0]

    if confile:
        confile = baseDir + '/' + confile

    # subprocess.getstatusoutput("rm -rf *")

    rc = ''
    args = ['coderrect']
    if not confile:
        if not lib:
            args += ["clang++ "+filepath+" -o " +
                     executableName + " -lpthread"]
            # args = ['ls']
            # workDir=os.getcwd()
            # outDir = pathlib.Path(workDir+'/results/')
            # stdoutDir = open(outDir.joinpath('{}-out.txt'.format(executableName)),'w')
            # stderrDir = open(outDir.joinpath('{}-err.txt'.format(executableName)), 'w')
            # print("args: {}\t outDir: {}\t stdoutDir: {}\t stderrDir: {}\t\n".format(args,outDir,stdoutDir,stderrDir))
            # subprocess.run(args, stdout=stdoutDir,
            # stderr=stderrDir,
            # cwd=workDir,timeout=60)
            # rc = subprocess.getstatusoutput(args)

        else:
            args += [baseDir+"/runbuildlib "+lib+" " + filepath]
    else:
        if not lib:
            args += ["-conf="+confile]+["clang++ " + filepath +
                                        " -o " + executableName + " -lpthread"]
            # rc = subprocess.getstatusoutput("coderrect -conf=" + confile + " clang++ " + filepath + " -o " + executableName + " -lpthread")
        else:
            args += ["-conf="+confile] + \
                [baseDir+"/runbuildlib " + lib + " " + filepath]

    stdoutDir = open(path.join(workDir,
                               '{}-out.txt'.format(executableName)), 'w')
    stderrDir = open(path.join(workDir,
                               '{}-err.txt'.format(executableName)), 'w')
    # print("args: {}\t workDir: {}\t stdoutDir: {}\t stderrDir: {}\t\n".format(
    #     args, workDir, stdoutDir.name, stderrDir.name))

    subprocess.run(args, stdout=stdoutDir, stderr=stderrDir,
                   cwd=workDir, timeout=60)

    # runcov
    covargs = [baseDir+'/runcov']
    #print('cov: {}'.format(covargs))
    # subprocess.getoutput(covargs)
    subprocess.run(covargs, stdout=stdoutDir, stderr=stderrDir,
                   cwd=workDir, timeout=60)

    returnstring = ''

    # check raw_$executableName.json
    if lib:
        jsonPath = path.join(
            workDir, ".coderrect/build/raw_{}.json".format(lib))
    else:
        jsonPath = path.join(workDir,
                             ".coderrect/build/raw_{}.json".format(executableName))

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
            returnstring += 'Found ' + rc[1] + \
                ' data races, expected ' + races + '\n'
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
        returnstring += 'Error: Could not find raw json: ' + jsonPath + '\n'
        return returnstring.rstrip()

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
            returnstring += 'Found ' + \
                rc[1] + ' deadlocks, expected ' + deadlocks + '\n'
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
        rc = subprocess.getstatusoutput(
            "jq '.mismatchedAPIs | length' " + jsonPath)
        rc = list(rc)
        rc[1] = rc[1].rstrip()
        if rc[0]:
            returnstring += rc[1] + '\n'
            fMismatchedAPI += int(mismatchedAPI)
            cmMismatchedAPI.fn += int(mismatchedAPI)
        if not rc[1] == mismatchedAPI:
            returnstring += 'Found ' + \
                rc[1] + ' mismatchedAPIs, expected ' + mismatchedAPI + '\n'
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
        rc = subprocess.getstatusoutput(
            "jq '.orderViolations | length' " + jsonPath)
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
    else:
        returnstring += 'Error: Could not find raw json' + '\n'

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
            returnstring += 'Found ' + rc[1] + \
                ' TOCTOUs, expected ' + toctou + '\n'
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
    else:
        returnstring += 'Error: Could not find raw json' + '\n'

    return returnstring.rstrip()


def workOnCase(pbar, case, baseDir, buildDir):
    import time
    start = time.time()

    filepath = baseDir+'/' + case
    (name, purpose, races, deadlocks, toctou, ov,
     mismatchedAPI, confile, lib) = parseCase(filepath)
    if (not purpose):
        #print('\n' + name + ' - invalid case')
        # print('\n----------------------------------------\n')
        end = time.time()
        return ['invalid', case, round(end-start)]

    workDir = path.join(buildDir, 'build-'+name)
    # create the work directory
    try:
        os.mkdir(workDir)
    except IOError as e:
        return ["Failed to create work directory - " + e, case]

    # go to workdir
    try:
        os.chdir(workDir)
    except IOError as e:
        print("Failed to change the working directory - " + e)

    #print('Test ' + name + ' - ' + purpose)
    errormsg = runRase(filepath, baseDir, workDir, races, deadlocks, toctou,
                       ov, mismatchedAPI, confile, lib)
    end = time.time()
    return [errormsg, case, round(end-start)]


class DetectResult:
    def __init__(self, status, msg, time):
        import pathlib
        self.status = status
        self.msg = msg
        self.time = time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def caculateCoveredFunctions(buildDir, case):
    name = case.split(".")[0]
    covFilePath = path.join(buildDir,
                            "build-{}/t.lcov.o".format(name))
    if os.path.isfile(covFilePath):
        arg = 'cat '+covFilePath + ' | wc -l'
        #print('arg: {}'.format(arg))
        covFunc = subprocess.getoutput(arg)
        return covFunc
    else:
        #print("covFilePath does not exist: {}".format(covFilePath))
        return 0

# main()


# get a list of cases
cases = getCases()

# create the working directory
baseDir = os.getcwd()
build = 'build'
build_last_file_path = ''
if path.exists(build):
    # move build to build_last
    count = subprocess.getoutput('ls | grep build_ | wc -l')
    idx = int(count)+1
    build_last_file_path = 'build_'+str(idx)
    if path.exists(build_last_file_path):
        try:
            # os.remove(build_last_file_path)
            import shutil
            shutil.rmtree(build_last_file_path)
        except OSError as e:
            print("Error: %s : %s" % (build_last_file_path, e.strerror))
    os.rename('build', build_last_file_path)

buildDir = path.join(baseDir, build)

try:
    os.mkdir(buildDir)
except IOError as e:
    print("Failed to create 'build' directory - " + e)
try:
    os.chdir(buildDir)
except IOError as e:
    print("Failed to change the working directory - " + e)

nFailed = 0
nInvalid = 0
nSucceeded = 0
successCase = ''
failCase = ''
summary = {}
parallel = multiprocessing.cpu_count()
print("parallel level: {}\n".format(parallel))
pbar = tqdm(total=len(cases), leave=True, position=0)
with ThreadPoolExecutor(max_workers=parallel) as executor:
    task_futures = [executor.submit(lambda p: workOnCase(
        pbar, p, baseDir, buildDir), case) for case in cases]
    for task_future in as_completed(task_futures):
        try:
            result = task_future.result()
            # print('\nResult: ' + result)
        except Exception as exc:
            print(
                f"Exception error: {exc}")
        else:
            # success/failure conditional
            status = 'p'
            msg = result[0]
            testcase = result[1]
            if result[0]:
                if result[0] == 'invalid':
                    nInvalid = nInvalid + 1
                    status = 'i'
                    #print('\nInvalid: ' + result[1])
                else:
                    nFailed = nFailed + 1
                    msg = 'failed: '+msg.replace('\n', '; ')
                    status = 'f'
                    failCase = testcase
                    #print('\nFailed: ' + result[0]+' on '+result[1])
            else:
                nSucceeded = nSucceeded + 1
                msg = 'success'
                successCase = testcase
                #print('\nSucceeded: '+result[1])
            rds = DetectResult(status, msg, result[2])
            summary[testcase] = rds
            pbar.write(
                f"{datetime.now().strftime('[%H:%M:%S]')}Done {testcase} '{msg}' ")
            pbar.update(1)
pbar.close()

# for case in cases:
print('\n----------------------------------------\n')
id = 0

# caculate coverage
# find any racedetect.lcov
casePath = ''
# cat racedetect.lcov | grep "FNF:" |  cut -f2 -d':' | awk '{for(i=1;i<=NF;i++)s+=$i}END{print s}'
if successCase:
    casePath = buildDir+'/build-'+successCase.split(".")[0]
    #print('totalFunc: {}'.format(totalFunc))
elif failCase:
    casePath = buildDir+'/build-'+failCase.split(".")[0]

if casePath:
    covarg = 'cat '+casePath+'/racedetect.lcov ' + \
        "| grep 'FNF:' |  cut -f2 -d':' | sort | uniq | awk '{for(i=1;i<=NF;i++)s+=$i}END{print s}'"
    totalFunc = subprocess.getoutput(covarg)
else:
    totalFunc = ''

summaryFileName = path.join(buildDir, 'summary.txt')
with pathlib.Path(summaryFileName).open('a') as summaryFile:
    for case in sorted(summary):
        id = id+1
        rds = summary[case]
        xyz = "{} '{}'".format(case, rds.status)
        summaryFile.write(xyz+'\n')
        covFunc = caculateCoveredFunctions(buildDir, case)
        if totalFunc:
            coverage = float(covFunc)/int(totalFunc)
            if coverage > 1:
                coverage = 1
            coverage = "{:.1%}".format(coverage)
        else:
            coverage = ''
        #print('coverage: {}'.format(coverage))
        printStr = "{}: {} {}s '{}' coverage: {}/{} ({})".format(
            id, xyz, str(rds.time), rds.msg, covFunc, totalFunc, coverage)
        if rds.status == 'p':
            print(printStr)
        elif rds.status == 'f':
            print(bcolors.FAIL+printStr+bcolors.ENDC)
        else:
            print(bcolors.WARNING+printStr+bcolors.ENDC)


print('\n----------------------------------------\n')
print('Total:'+str(nSucceeded+nFailed)+'; Succeeded: ' +
      str(nSucceeded)+'; Failed: '+str(nFailed)+'\n')

# Total: 87; Succeed: 73; Failed: 14
if build_last_file_path:
    # check regression
    buildLastDir = path.join(baseDir, build_last_file_path)
    summaryFile_last = path.join(buildLastDir, 'summary.txt')
    args = 'diff ' + summaryFileName + ' ' + summaryFile_last
    print('regression: {}'.format(args))
    diff = subprocess.getoutput(args)
    print(diff)

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
