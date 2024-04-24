#This script is used to compare benchmark testing result with previous running,
#it would be easy to find out if new cases added, or bug introduced.

import re
import sys

def getCaseName(record):
    ret = re.search('Test (\\S+) -', record)
    if ret is not None:
        return ret.group(1)
    else:
        return None

def getCaseStatus(record):
    ret = re.search('Succeed', record)
    if ret is not None:
        return 'Succeed'
    else:
        return 'Failed'

def getFileContent(file):
    with open(file, 'r') as f:
        r = f.read()
    return r

def getBaseRecords(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    baseRecord = {}
    for l in lines:
        r = l.strip('\n').split(' ')
        baseRecord[r[0]] = r[1]
    return baseRecord

def parseRecords(text):
    records = text.split('----------------------------------------')
    list = []
    for r in records:
        n = getCaseName(r)
        s = getCaseStatus(r)
        if n is not None:
            list.append((n, s))
    return list

def compareRecords(baseFile, targetFile):
    d_baseRecords = getBaseRecords(baseFile)
    newRecords = parseRecords(getFileContent(targetFile))

    cases_without_change = []
    cases_with_change = []
    cases_new = []

    for record in newRecords:
        case = record[0]
        status = record[1]
        if case in d_baseRecords.keys():
            if status == d_baseRecords[case]:
                compareResult = ' '
                cases_without_change.append((case, d_baseRecords[case], status, compareResult))
            else:
                compareResult = 'changed'
                cases_with_change.append((case, d_baseRecords[case], status, compareResult))
        else:
            cases_new.append((case, ' ', status, 'new'))
    final = cases_new + cases_with_change + cases_without_change
    return final

def printRecords():
    c = getFileContent('x')
    records = parseRecords(c)
    for r in records:
        print(r[0] + ' ' + r[1])

def usage():
    print("Usage: python3 compareBenchmark.py basefile targetfile")
    print("       #basefile - benchmark tracking file, now it 's benchmark.track.")
    print("       #targetfile - log file of benchmark testing, now it 's at_benchmark.log.")
    print("       #e.g. python3 compareBenchmark.py benchmark.track at_benchmark.log")

def main():
    if len(sys.argv) != 3:
        usage()
        exit(1)
    baseFile = sys.argv[1]
    targetFile = sys.argv[2]
    result = compareRecords(baseFile, targetFile)
    print('{0:40} {1:10} {2:10} {3:10}'.format('Case Name', 'Expected', 'Actual', 'Status'))
    for r in result:
        print('{0:40} {1:10} {2:10} {3:10}'.format(r[0], r[1], r[2], r[3]))

if __name__ == '__main__':
    main()

