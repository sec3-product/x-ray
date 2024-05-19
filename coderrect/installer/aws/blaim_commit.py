import os
import json
import time
import re
import getopt
import sys

repos = ['LLVMRace', 'gllvm', 'executor', 'reporter']
bucket_template = 's3://cr-qa-automation/logs/installer/BRANCH/'
#logFile_template = 'commit_BRANCH_' + time.strftime("%Y%m%d%H%M", time.localtime()) + '.log'
logFile_template = 'commit_BRANCH_T.log'  # file name will be changed by CI/CD system, so don't attach timestamp here.
s3file = None
no_new_commit = 'no new commit'
cmd_last_logFile = 'aws s3 ls BUCKET | awk \'{print $4}\' | grep commit_BRANCH | sort | tail -1'
cmd_download_file = 'aws s3 cp FILE ./'
cmd_upload_file = 'aws s3 cp FILE BUCKET'
cmd_latestCommit = 'cd REPO_DIR && git log --oneline -1 | cut -d" " -f 1'
cmd_showCommit = 'cd REPO_DIR && git --no-pager log --date=short --pretty=format:"%h %cn %ci %s" -2000'
cmd_currentBranch = 'cd REPO_DIR && git branch | grep \'*\''
last_commit = {}
config={}

def isRepoExist(repo):
    return os.path.exists(repo)

def validateAllReposExist():
    ret = True
    for repo in repos:
        ret = ret and isRepoExist(repo)
    return ret

def getCurrentBranch(repo):
    cmd = cmd_currentBranch.replace('REPO_DIR', repo)
    r = os.popen(cmd)
    t = r.read()

    ret = re.search('\* (\S+)', t)
    if ret is not None:
        return ret.group(1)
    return None

def getCurrentTagOnMaster(repo):
    cmd = cmd_currentBranch.replace('REPO_DIR', repo)
    r = os.popen(cmd)
    t = r.read()

    ret = re.search('\* \(HEAD detached at (\S+)\)', t)
    if ret is not None:
        return ret.group(1)
    return None

def validateCheckout(repo):
    if 'master' == config['brCode']:
        current = getCurrentTagOnMaster(repo)
    else:
        current = getCurrentBranch(repo)
    return current == config['checkout']

def validateCheckoutForAll():
    ret = True
    for repo in repos:
        ret = ret and validateCheckout(repo)
    return ret

def getLatestCommit(repo):
    c = cmd_latestCommit.replace('REPO_DIR', repo)
    r = os.popen(c)
    commit = r.read()
    return commit.strip('\n')


def getAddedCommit(repo):
    lastCommit = last_commit[repo]
    if lastCommit == getLatestCommit(repo):
        return no_new_commit

    l = []
    cmd = cmd_showCommit.replace('REPO_DIR', repo)
    r = os.popen(cmd)
    list = r.readlines()

    for record in list:
        commit = record.split(' ', 1)[0]
        if commit == lastCommit:
            break
        else:
            l.append(record)
    return l


def generateFirstFile():
    if not validateAllReposExist():
        print('Some repository is not existing for the first comparing file creating, pls check it.')
        exit(1)
    if not validateCheckoutForAll():
        print('Some repository is not checkout on right branch or tag, pls check it.')
        exit(1)

    l = []
    for r in repos:
        d = {}
        d['repository'] = r
        d['checkout'] = config['checkout']
        d['latest_commit'] = getLatestCommit(r)
        d['added_commits'] = no_new_commit

        l.append(d)

    with open(config['logFile'], 'w') as f:
        data = json.dumps(l, indent=4)
        f.write(data)


def generateComparingFile():
    l = []

    for r in repos:
        d = {}
        d['repository'] = r
        d['checkout'] = config['checkout']

        if isRepoExist(r):
            if validateCheckout(r):
                d['latest_commit'] = getLatestCommit(r)
                d['added_commits'] = getAddedCommit(r)
            else:
                print('Repository ' + r + ' is checkout on incorrect branch or tag, which should not happen, pls check.')
                exit(1)
        else:
            d['latest_commit'] = last_commit[r]
            d['added_commits'] = no_new_commit
        d['last_commit'] = last_commit[r]
        l.append(d)

    with open(config['logFile'], 'w') as f:
        data = json.dumps(l, indent=4)
        f.write(data)

def getS3LogFile():
    cmd = cmd_last_logFile.replace('BUCKET', config['bucket']).replace('BRANCH', config['version'])
    r = os.popen(cmd)
    t = r.read()
    print('Last commit comparing file: ' + t)

    if t != "":
        global s3file
        s3file = t.strip('\n')
        cmd = cmd_download_file.replace('FILE', config['bucket'] + s3file)
        os.system(cmd)

def uploadLogFile(file):
    cmd = cmd_upload_file.replace('FILE', file).replace('BUCKET', config['bucket'])
    os.system(cmd)

def getLastCommit():
    with open(s3file, 'r') as f:
        t = f.read()
    list = json.loads(t)
    for repo in list:
        last_commit[repo['repository']] = repo['latest_commit']

def parse_args(argv):
    try:
        opts, args = getopt.getopt(argv, "-r:")
    except getopt.GetoptError:
        usage()
        exit(1)
    if len(opts) == 1:
        opt, arg = opts[0]
        config['checkout'] = 'release-v' + arg
        config['brCode'] = 'release'
        config['version'] = arg
    elif len(args) == 1:
        config['checkout'] = 'v' + args[0]
        config['brCode'] = 'master'
        config['version'] = args[0]
    else:
        usage()
        exit(1)
    config['bucket'] = bucket_template.replace('BRANCH', config['brCode'])
    config['logFile'] = logFile_template.replace('BRANCH', config['version'])

def usage():
    print('Usage:')
    print('python3 blaim_commit.py -r 0.1.0 #For release branch')
    print('python3 blaim_commit.py 0.1.0    #For master branch')

def main():
    parse_args(sys.argv[1:])
    getS3LogFile()
    if s3file is None:
        generateFirstFile()
    else:
        getLastCommit()
        generateComparingFile()
    #uploadLogFile(config['logFile'])   # CI/CD system will upload file after changing file name with timestamp.

if __name__ == '__main__':
    main()
