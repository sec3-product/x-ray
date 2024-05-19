#######################################################################
#This script is used to trigger jenkins job on aws instance.
#Usage:  python3 trigger_Jenkins_job.py job_name '{"version":"0.3.0"}'
#######################################################################
import time
import jenkins
import sys
import json

user = "michael"
token = "11a5adab477bf011cd98cdcf2f664bd02b"
url = 'http://127.0.0.1:9000/jenkins'
job = sys.argv[1]
param = sys.argv[2]

def triggerBuild(url, job, **kwargs):
    server = jenkins.Jenkins(url, username=user, password=token, timeout=10)
    server.get_whoami()

    next_build_number = server.get_job_info(job)['nextBuildNumber']
    print("next build number is: " + str(next_build_number) + ", trigger it.")

    #server.build_job(job, {'version': version})
    server.build_job(job, kwargs)
    time.sleep(10)

    last_build_number = server.get_job_info(job)['lastBuild']['number']
    print("last build number is: " + str(last_build_number) + ".")

    if (last_build_number == next_build_number):
        print("Job " + job + " is triggered successfully!")
    else:
        print("Job " + job + " isn't triggered correctly, pls check!")
        return -1

    return last_build_number


def main():
    buildParams = json.loads(param)
    #timestamp will be generated automatically for scheduled tasks.
    buildParams['timestamp'] = time.strftime("%Y%m%d%H%M", time.localtime())

    buildNum = triggerBuild(url, job, **buildParams)
    if buildNum != -1:
        result = "Build No #" + str(buildNum) + " for " + job + " is trigged successfully"
        print(result)
    else:
        print("Failed to trigger job :" + job)

if __name__ == '__main__':
    main()
