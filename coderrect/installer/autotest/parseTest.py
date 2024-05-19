import re
import sys

outputFile = 'parseResult.file'
pattern_only_failed = '=+ (\\d+) failed, '
pattern_only_passed = '=+ (\\d+) passed, '
pattern_failed_passed = '=+ (\\d+) failed, (\\d+) passed'
pattern_final_result = '=+ (.*) \(\\d+:\\d+:\\d+\) =+'

def getTestSummary():
    inputFile = sys.argv[1]
    with open(inputFile, 'r') as f:
        text = f.read()
    return text

def getTestElapse(summary):
    ret = re.search('=+ .* \((.*)\) =+', summary)
    return ret.group(1)

def getStatsData(summary):
    if re.search(pattern_failed_passed, summary) is not None:
        ret = re.search(pattern_failed_passed, summary)
        failed = int(ret.group(1))
        passed = int(ret.group(2))
    elif re.search(pattern_only_failed, summary) is not None:
        ret = re.search(pattern_only_failed, summary)
        failed = int(ret.group(1))
        passed = 0
    elif re.search(pattern_only_passed, summary) is not None:
        ret = re.search(pattern_only_passed, summary)
        failed = 0
        passed = int(ret.group(1))
    return failed, passed

def getFinalResult(summary):
    ret = re.search(pattern_final_result, summary)
    if ret is not None:
        return ret.group(1)
    else:
        return 'No result found.'

def parse():
    text = getTestSummary()
    failed, passed = getStatsData(text)
    total = failed + passed
    elapse = getTestElapse(text)

    with open(outputFile, 'w') as f:
        f.write("total=" + str(total) + "\n")
        f.write("failed=" + str(failed) + "\n")
        f.write("success=" + str(passed) + "\n")
        f.write("test_elapse=" + elapse + "\n")

def parseFinlaResult():
    text = getTestSummary()
    r = getFinalResult(text)

    with open(outputFile, 'w') as f:
        f.write("finalResult=" + r + "\n")

def usage():
    print("Usage: python3 parseTest.py test-summary-file")

def main():
    if len(sys.argv) != 2:
        usage()
        exit(1)
    parseFinlaResult()

if __name__ == '__main__':
    main()
