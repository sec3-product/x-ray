#!/usr/bin/python3

# Name of binary must be in PATH
TOOL='racedetect'

# Assumes current directory is git repo at the built tool commit
def get_tool_version():
    import subprocess
    return subprocess.check_output(["git", "describe"]).strip().decode('utf-8')


def test_bench(path_to_bc):
    import os
    import subprocess

    path_to_bc = os.path.abspath(path_to_bc)
    if not os.path.exists(path_to_bc):
        print('Could not find {}'.format(path_to_bc))
        return None

    base = os.path.basename(path_to_bc)
    bench_name = os.path.splitext(base)[0]

    print('Running on {}'.format(base))
    try:
        cmd = TOOL + ' -o {}-{}-races.json '.format(bench_name, get_tool_version())  + path_to_bc
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    except subprocess.CalledProcessError as e:
        print('Tool failed to run on {}'.format(bench_name))
        print(e.output.decode('utf-8'))
        print(e.returncode)
        return
    

    with open('{}-{}-report.txt'.format(bench_name, get_tool_version()), 'w') as f:
        f.write(output)


help_msg = '''
Expected path(s) to .bc files of benchmarks
Example:
    $ ./test-benches.py LULESH/lulesh2.0.bc PI/pi.bc
'''


if __name__ == '__main__':
    import sys
    import os
    BENCHMARKS = sys.argv[1:]

    if len(BENCHMARKS) == 0:
        print(help_msg)
        sys.exit(0)
    
    for bench in BENCHMARKS:
        if not os.path.exists(bench):
            print('Could not find {}'.format(bench))
            print(help_msg)
            sys.exit(1)

    for bench in BENCHMARKS:
        test_bench(bench)
        