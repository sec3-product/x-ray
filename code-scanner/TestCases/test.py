#!/usr/bin/python3

import sys
import os


######## Potentially Edit These #######

# Name of binary tool to run
TOOL='racedetect'


############ Begin Program ###########

ROOT = os.path.abspath(os.getcwd())
WORKING_DIR = os.path.join(ROOT, '.testworking')
GLOBAL_FLAGS = ' -fopenmp -O1 -g -fno-discard-value-names ' 

def is_bin_in_path(bin):
    from shutil import which
    return which(bin) != None

# Check for all dependencies and create working directory
def setup():
    import sys
    from os import mkdir, chdir
    from shutil import rmtree

    bins=[TOOL]
    for b in bins:
        if not is_bin_in_path(b):
            print('Could not find', b, 'in PATH')
            sys.exit(1)

    if os.path.isdir(WORKING_DIR):
        rmtree(WORKING_DIR)
        
    mkdir(WORKING_DIR)

# Delete all intermediate files
def cleanup():
    from shutil import rmtree

    os.chdir(ROOT)
    rmtree(WORKING_DIR)

# Run tool and return stdout/stderr as string
def run_tool(bc_path):
    import subprocess
    if not os.path.isabs(bc_path):
            raise ValueError('not absolute path')

    cmd = TOOL + ' ' + bc_path + ' --no-filter'
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=WORKING_DIR).decode('utf-8')
    except subprocess.CalledProcessError as e:
        # TODO: add flag to print this conditionally
        #print(e.output.decode('utf-8'))
        #print(e.returncode)
        return None

# Find all line/column pairs from tool output
def parse_tool_output(output):
    import re
    races = []
    for line in output.split('\n'):
        race = re.findall(r'line (\d+), column (\d+)', line)
        if race:
            races.append(race)
    return races

# return set of line numbers present in races
# races should be list of races where each race is a list of (line,col) locations
def get_race_lines(races):
    lines = set()
    for race in races:
        for loc in race:
            lines.add(str(loc[0]))
    return lines

# run the tool on a bc file and check if the output matches the expected result
#   bc_path: absolute path to .bc/.ll file
#   expected_result: list of races where each race is a list of (col,line) that participate in race
# returns (RESULT, NUM_RACES, TIME, EXTRA_OUTPUT, TOOL_OUTPUT)
def run_test(bc_path, expected_result):
    from time import time

    start = time()
    output = run_tool(bc_path)
    end = time()

    if output is None:
        return ('CRASH', '', '', '', '')
    
    race_count = str(output.count('Found a race')) + ' races'
    actual_races = parse_tool_output(output)

    actual_race_lines = get_race_lines(actual_races)
    expected_races_lines = get_race_lines(expected_result)
    
    sucess = actual_race_lines == expected_races_lines
    status = 'PASS' if sucess else 'FAIL'
    runtime = '{0:.2f}'.format(end-start) + 's'
    extra = None
    if not sucess:
        extra = 'Expected lines: {}, Got: {}'.format(expected_races_lines, actual_race_lines)

    return (status, race_count, runtime, extra, output)

# Build a single c or cpp file and return a path to the resulting bc file
def build_c(cfile, flags='', lflags='', logoutput=False, savebc=False):
    import subprocess
    from shutil import copy

    if not os.path.isabs(cfile):
        raise ValueError('not absolute path')
    
    CC = 'clang ' if cfile.endswith('.c') else 'clang++ '
    
    flags += GLOBAL_FLAGS + ' -S -emit-llvm '
    filename = os.path.basename(cfile).split('.')[0]
    outfile = os.path.join(WORKING_DIR, filename + '.ll')

    cmd = CC + flags + cfile + ' -o ' + outfile + ' ' + lflags
    out = sys.stdout if logoutput else subprocess.DEVNULL
    try:
        subprocess.run(cmd, shell=True, stdout=out, stderr=out, cwd=WORKING_DIR)
    except subprocess.CalledProcessError as e:
        # TODO: add flag to print this conditionally
        print(e.output)
        print(e.returncode)
        return None
    
    # Check that file was built
    if not os.path.isfile(outfile):
        return None
    
    if savebc:
        copy(outfile, savebc)

    return outfile

# return dict of expected results from json file
def load_expected(filename):
    import json
    json_data = open(filename).read()
    return json.loads(json_data)

# Build, test, and report on all tests in dataracebench
#   enabling showMore prints extra information about failing cases
def test_DRB(showMore=False, savebc=False, match=None):
    import subprocess
    from glob import glob
    from time import time
    from pprint import pprint

    # Load race information
    expected_results = load_expected('drbexpect.json')

    SOURCE_PATH=os.path.join(ROOT, "dataracebench/micro-benchmarks")

    CFLAGS="-g -O1 -std=c99 -fopenmp -fno-discard-value-names"
    CPPFLAGS="-g -O1 -fopenmp -fno-discard-value-names"
    POLYFLAG="{0}/utilities/polybench.c -I {0} -I {0}/utilities -DPOLYBENCH_NO_FLUSH_CACHE -DPOLYBENCH_TIME -D_POSIX_C_SOURCE=200112L".format(SOURCE_PATH)

    cFiles = list(glob('{}/*.c'.format(SOURCE_PATH)))
    cFiles += list(glob('{}/*.cpp'.format(SOURCE_PATH)))
    cFiles.sort()

    passing_cases = []
    failing_cases = []
    crashing_cases = []

    name_col_width = max([len(n) for n in expected_results.keys()]) + 2

    for f in cFiles:
        filename = os.path.basename(f).split('.')[0]

        if match and match not in filename:
            continue

        flags = CFLAGS if f.endswith('.c') else CPPFLAGS
        flags += POLYFLAG if 'PolyBench' in f else ''
        bc = build_c(f, flags=flags, savebc=savebc)
        if not bc:
            print(filename.ljust(name_col_width), 'SKIP')
            continue
        
        expected_result = []
        if filename.endswith('yes'):
            expected_result = expected_results[filename]['races']
        
        status, race_count, runtime, extra, output = run_test(bc, expected_result)

        print(filename.ljust(name_col_width), status, '\t', race_count, '\t', runtime)
        if showMore and extra:
            print(extra)

        if status == 'PASS':
            passing_cases.append(filename)
        elif status == 'FAIL':
            failing_cases.append(filename)
        elif status == 'CRASH':
            crashing_cases.append(filename)
        else:
            raise Exception('Unhandled test result')

    def get_count(l, s):
        return len([x for x in l if x.endswith(s)])
    
    yes_pass = get_count(passing_cases, 'yes')
    yes_fail = get_count(failing_cases, 'yes')
    yes_crash = get_count(crashing_cases, 'yes')

    no_pass = get_count(passing_cases, 'no')
    no_fail = get_count(failing_cases, 'no')
    no_crash = get_count(crashing_cases, 'no')

    print('PASS/FAIL/CRASH')
    print('Total:\t', len(passing_cases), '/', len(failing_cases),  '/', len(crashing_cases))
    print('yes:\t', yes_pass, '/', yes_fail, '/', yes_crash)
    print('no:\t', no_pass, '/', no_fail, '/', no_crash)

def test_regression(showMore=False, savebc=False, match=None):
    from glob import glob
    from time import time

    src=os.path.join(ROOT, "regression")
    expected_results = load_expected(os.path.join(src, 'expected_results.json'))

    cFiles = list(glob('{}/*.c'.format(src)))
    cFiles += list(glob('{}/*.cpp'.format(src)))
    cFiles.sort()

    passing_cases = []
    failing_cases = []
    crashing_cases = []
    name_col_width = max([len(n) for n in expected_results.keys()]) + 2

    for f in cFiles:
        filename = os.path.basename(f).split('.')[0]
        if filename not in expected_results:
            continue

        if match and match not in filename:
            continue
        
        bc = build_c(f, savebc=savebc)
        if not bc:
            print(filename.ljust(name_col_width), 'SKIP')
            continue
        
        status, race_count, runtime, extra, output = run_test(bc, expected_results[filename]['races'])

        print(filename.ljust(name_col_width), status, '\t', race_count, '\t', runtime)
        if showMore and extra:
            print(extra)

        if status == 'PASS':
            passing_cases.append(filename)
        elif status == 'FAIL':
            failing_cases.append(filename)
        elif status == 'CRASH':
            crashing_cases.append(filename)
        else:
            raise Exception('Unhandled test result')

    def get_count(l, s):
        return len([x for x in l if x.endswith(s)])
    
    yes_pass = get_count(passing_cases, 'yes')
    yes_fail = get_count(failing_cases, 'yes')
    yes_crash = get_count(crashing_cases, 'yes')

    no_pass = get_count(passing_cases, 'no')
    no_fail = get_count(failing_cases, 'no')
    no_crash = get_count(crashing_cases, 'no')

    print('PASS/FAIL/CRASH')
    print('Total:\t', len(passing_cases), '/', len(failing_cases),  '/', len(crashing_cases))
    print('yes:\t', yes_pass, '/', yes_fail, '/', yes_crash)
    print('no:\t', no_pass, '/', no_fail, '/', no_crash)


# Attempt to build and test .c/.cpp files or just test .ll/.bc files
#  prints results to stdout
def test_single(filepath, savebc=False):
    if not os.path.isfile(filepath):
        print(filepath, 'is not a file!')
        return
    
    if filepath.endswith('.c'):
        bc = build_c(filepath, savebc=savebc)
        if bc:
            print(run_tool(bc))
    elif filepath.endswith('.cpp'):
        pass
    elif filepath.endswith('.ll') or filepath.endswith('.bc'):
        print(run_tool(filepath))
    else:
        print('Unknown file type!')

def main(args):
    if args.single:
        abs_path = os.path.join(ROOT, args.single)
        test_single(abs_path, savebc=args.save_bc)
        return
    if args.regression:
        test_regression(showMore=args.verbose, savebc=args.save_bc, match=args.match)
        return
    test_DRB(showMore=args.verbose, savebc=args.save_bc, match=args.match)
    

if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--single', '-s', metavar='FILE', 
                    type=str, help='Test tool on a single file')
    parser.add_argument("--save-bc", metavar='DIR',
                    help="dumps any .bc files built to current directory",
                    type=str)
    parser.add_argument("--verbose", '-v', help="increase output verbosity",
                    action="store_true")
    parser.add_argument("--regression", '-r', help="run tets in regression folder",
                    action="store_true")
    parser.add_argument("--match", '-m', metavar='substr',
                    help="Only run tests containing some substr",
                    type=str)

    args = parser.parse_args()

    if args.save_bc:
        args.save_bc = os.path.abspath(args.save_bc)
        if not os.path.exists(args.save_bc):
            os.mkdir(args.save_bc)


    setup()
    try:
        main(args)
    finally:
        cleanup()


        
