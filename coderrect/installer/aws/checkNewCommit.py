import os
import sys
import getopt

cmd_git_clone="git clone git@github.com:coderrect-inc/REPOSITORY.git"
cmd_git_log="cd REPOSITORY && git checkout BRANCH && git log --since='COUNT hours ago' --pretty=oneline --abbrev-commit"
repos=['LLVMRace', 'gllvm', 'executor', 'reporter']

def downloadRepository(repo):
    if repo == 'LLVMRace':
        cmd = 'git clone git@github.com:coderrect/LLVMRace.git'
    else:
        cmd = cmd_git_clone.replace('REPOSITORY', repo)
    os.system(cmd)

def isRepoUpdatedRecently(repo, branch, count):
    downloadRepository(repo)
    cmd = cmd_git_log.replace('REPOSITORY', repo).replace('BRANCH', branch).replace('COUNT', count)
    with os.popen(cmd) as f:
        r = f.readlines()
    print(r)
    if len(r) > 1:
        return True
    return False

def isBuildNeeded(branch, count):
    for repo in repos:
        if isRepoUpdatedRecently(repo, branch, count):
            print('Found new commit in repository ' + repo + ', new build will be triggered.')
            return True
    return False

def usage():
    print('Usage: python3 checkNewCommit.py <-d | -r <version>> -c <hour count>')
    print('       -d compare on develop branch')
    print('       -r compare on release branch')
    print('       -c hours ago you want to compare with')
    print('       -h help message')
    print('       e.g. python3 checkNewCommit.py -r 0.5.0 -h 24')

def parse_args(argv):
    try:
        opts, args = getopt.getopt(argv, "dhr:c:")
    except getopt.GetoptError:
        usage()
        exit(1)

    branch = None
    count = None
    for opt, arg in opts:
        if opt == '-h':
            usage()
            exit(0)
        elif opt == '-d':
            branch = 'develop'
        elif opt == '-r':
            branch = 'release-v' + arg
        elif opt == '-c':
            count = arg
    if branch is None or count is None:
        usage()
        exit(1)
    return branch, count

def main():
    branch, count = parse_args(sys.argv[1:])

    r = isBuildNeeded(branch, count)
    print('needBuild=' + str(r))

if __name__ == '__main__':
    main()
