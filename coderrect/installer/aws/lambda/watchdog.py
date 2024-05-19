import time
import boto3
import json

ZONE = 'us-west-2'
KEY_ID = 'AKIAZS62V2IXWAPEGFG7'
ACCESS_KEY = 'C5fvrxbL7MC2yIz3c40hPhWHzZLAcL7pT5/qYJxH'

client = boto3.client(
    'ec2',
    region_name=ZONE,
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=ACCESS_KEY)

def getInstances():
    response = client.describe_instances(
        Filters=[
            {
                'Name': 'instance-state-name',
                'Values': [
                    'running',
                ]
            },
        ],
    )
    if response is not None:
        return response['Reservations']
    return None

def trans2timestamp(lauchTime):
    t = lauchTime.split('+', 1)[0]
    ta = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(ta))
    return ts

def getCurrentTimestamp():
    t = time.time()
    return int(t)

def isEc2ExistsLongTime(launchTime):
    launchOn = trans2timestamp(launchTime)
    current = getCurrentTimestamp()

    if current - launchOn > 2400:
        return True
    return False

def termInstance(instanceId):
    try:
        response = client.terminate_instances(
            InstanceIds=[
                instanceId,
            ]
        )
    except Exception as e:
        print(e)

    print(response)

def lambda_handler(event, context):
    list = getInstances()
    if list is None:
        print("No running instance was found.")
        exit(0)

    for item in list:
        ec2 = item['Instances'][0]
        instanceId = ec2['InstanceId']
        launchTime = str(ec2['LaunchTime'])
        if isEc2ExistsLongTime(launchTime):
            termInstance(instanceId)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
