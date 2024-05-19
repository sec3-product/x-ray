#/bin/bash

AMI=ami-056a9e5fdc3cd5125
type=t2.medium
keyfile=coderight
sgroup=sg-02bb5103a420825f8
subnet=subnet-65e00a1d
port=22
user=ubuntu
ready=false
time=$(date +%s)

if [ $# -lt 1 ];then
	echo "error: missing branch or version number"
	echo "usage: ./build_helper.sh -r 0.0.1    or"
	echo "       ./build_helper.sh 0.0.1"
	exit 1
fi

#start a new instance from AMI
aws ec2 run-instances \
	--image-id $AMI \
	--count 1 \
	--instance-type $type \
	--key-name $keyfile \
	--security-group-ids $sgroup \
	--subnet-id $subnet \
	--associate-public-ip-address \
	--tag-specifications "ResourceType=instance,Tags=[{Key=createAt,Value=$time}]"

#query the instance id and ip of new instance
echo "wait for instance(createAt=$time) to start....."
sleep 10
instanceId=$(aws ec2 describe-instances \
	--filters Name=image-id,Values=$AMI Name=tag:createAt,Values=$time \
	--query 'Reservations[*].Instances[*].[InstanceId]' \
	--output text)

host=$(aws ec2 describe-instances \
	--filters Name=image-id,Values=$AMI Name=tag:createAt,Values=$time \
	--query 'Reservations[*].Instances[*].[PublicIpAddress]' \
	--output text)

echo "instance: $instanceId"
echo "host: $host"
echo "wait for ssh service ready......"

#wait for ssh ready
for ((i=0; i<20; i++))
do
	result=$(nc -zvw3 $host $port 2>&1)
        echo $result
        if [[ "$result" =~ .*"succeeded".* ]];then
                echo "The remote ec2 instance is ready to connect."
		ready=true
                break
        fi
        sleep 10
done

#exit whole script if ssh can't be ready
if [ "$ready" = "false" ];then
	echo "SSH service is unavailable on remote ec2 instance for long time, please check it."
	exit 1
fi

#prepare one init script on remote instance
ssh -i $keyfile.pem -o StrictHostKeyChecking=no $user@$host 'cat > build.sh <<EOF
#/bin/bash

git clone git@github.com:coderrect-inc/installer.git
cd installer
. ~/.profile
./build-pkg \$@

#upload to s3 bucket
echo "uploading installer to s3 bucket......"
./upload-pkg
echo "finished"

#clean the directory
cd
sudo rm -rf installer/
EOF'

ssh -i $keyfile.pem -o StrictHostKeyChecking=no $user@$host 'chmod 755 build.sh'
ssh -t -i $keyfile.pem -o StrictHostKeyChecking=no $user@$host "./build.sh $@"

aws ec2 terminate-instances --instance-id $instanceId
