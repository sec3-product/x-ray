import time
import boto3
import json
import jenkins

type = "c5d.9xlarge"
keyfile = "coderight"
sgroup = "sg-02bb5103a420825f8"
subnet = "subnet-65e00a1d"

user = "michael"
token = "11a5adab477bf011cd98cdcf2f664bd02b"

ZONE = 'us-west-2'
KEY_ID = 'AKIAZS62V2IXWAPEGFG7'
ACCESS_KEY = 'C5fvrxbL7MC2yIz3c40hPhWHzZLAcL7pT5/qYJxH'

client = boto3.client(
    'ec2',
    region_name=ZONE,
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=ACCESS_KEY)


def newInstance(image):
    try:
        response = client.run_instances(
            ImageId=image,
            InstanceType=type,
            MaxCount=1,
            MinCount=1,
            KeyName=keyfile,
            SubnetId=subnet,
            SecurityGroupIds=[
                sgroup,
            ]
        )
        print(response)
        instanceId = response['Instances'][0]['InstanceId']

        print("wait for instance creating......")
        time.sleep(20)

        host = getIp(instanceId)
        print("instance id: " + instanceId)
        print("ip address: " + host)

    except Exception as e:
        print(e)

    return instanceId, host


def getIp(instanceId):
    response = client.describe_instances(
        InstanceIds=[
            instanceId,
        ]
    )
    ipv4 = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    return ipv4


def waitForJenkinsUp(url, repeat=6, interval=20):
    server = jenkins.Jenkins(url, username=user, password=token, timeout=10)

    #wait for port openning
    for count in range(repeat):
        try:
            server.get_whoami()
        except Exception as e:
            if (count >= repeat - 1):
                print("Server can't be started in sufficient time, pls check it")
                return False
            else:
                print("Server is starting, pls wait......")
                time.sleep(interval)
        else:
            print("Server is up.")
            break

    #wait for jenkins ready
    for count in range(repeat):
        try:
            server.wait_for_normal_op(interval)
            print("jenkins is ready to receive request.")
            ver = server.get_version()
            print("jenkins version: " + ver)
            break
        except Exception as e:
            if (count >= repeat - 1):
                print("Jenkins failed to be ready in sufficient time")
                return False
            else:
                print("Jenkins is starting, pls wait......")
    return True


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


def lambda_handler(event, context):
    d = json.loads(event["body"])
    ami = d["image"]
    job = d["job"]

    buildParams = d["params"]

    #timestamp will be generated automatically for scheduled tasks.
    if 'schedule' in d.keys():
        buildParams['timestamp'] = time.strftime("%Y%m%d%H%M", time.localtime())

    instanceId, host = newInstance(ami)
    url = 'http://' + host + ':9000/jenkins'

    result = "Build isn't triggered correctly, please check it!"
    if waitForJenkinsUp(url):
        buildNum = triggerBuild(url, job, **buildParams)
        if buildNum != -1:
            result = "Build No #" + str(buildNum) + " for " + job + " is trigged successfully"

    return {
        'statusCode': 200,
        'body': result
    }

