#/bin/bash

job=custom-clang-develop
image=ami-0d60232e09dc4357d
timestamp=$(date +%s)

s3_log_dir="s3://cr-qa-automation/logs/custom_clang/develop"
file=${job}_$timestamp.log

if [ $# -lt 1 ];then
	echo "error: missing params"
	echo "usage: $0 -d"
	exit 1
fi

#`curl https://2lczhzl702.execute-api.us-west-2.amazonaws.com/dev/build?version=$1\&timestamp=$timestamp > /dev/null 2>&1`

curl -H "Content-Type:application/json" -X POST -d "{\"image\":\"$image\",\"job\":\"$job\",\"params\":{\"timestamp\":\"$timestamp\"}}" https://2lczhzl702.execute-api.us-west-2.amazonaws.com/dev/build > /dev/null 2>&1

#wait for s3 log ready
result=""
for ((i=0; i<30; i++))
do
	result=$(aws s3 ls ${s3_log_dir}/${file}.gz)
	if [ "$result" != "" ];then
		echo "Build log is generated."
        	break
        fi
	
	echo "The building is ongoing, pls wait......"
	sleep 30
done

if [ "$result" == "" ];then
	echo "No reponse in long time, exit running, pls check it."
	exit 1
fi

#delay 5 seconds to avoiding it 's uploading
sleep 5
aws s3 cp ${s3_log_dir}/${file}.gz ./
gzip -d ${file}.gz
cat $file

echo "terminate instance......"
instanceId=$(grep instanceId $file | cut -d= -f 2)
aws ec2 terminate-instances --instance-id $instanceId
