#This script is used to merge release branch to master for the following repositories: LLVMRace, gllvm, executor, and reporter.

import json
import os
import sys

jsonFile = 'config.json'
def getConfig():
    try:
        with open(jsonFile, "r") as f:
            d = json.load(f)
    except Exception as e:
        print("Error happens when reading config file.")
        print(e)
        exit(1)
    return d

def executeCmd(cmd):
    r = os.system(cmd)
    if r != 0:
        print("Error happens when execute cmd: " + cmd)
        exit(1)

def usage():
    print("Usage:   python3 mergeBranch.py version  #e.g., python3 mergeBranch.py 0.1.0")

def main():
    if len(sys.argv) != 2:
        usage()
        exit(1)
    version = sys.argv[1]
    d = getConfig()
    for repo in d['Repos']:
        print('Start to merge branch in repository: ' + repo['name'])
        cmd = repo['cmd'].replace('VERSION', version)
        executeCmd(cmd)
    print("Merge finished successfully!")

if __name__ == '__main__':
    main()

