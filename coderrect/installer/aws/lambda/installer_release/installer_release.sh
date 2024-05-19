#/bin/bash

job=build-installer-release
image=ami-090aa7c7e8371f737
file_prefix=build-installer-release
summary_prefix=summary
report_prefix=report
timestamp=$(date +%s)

s3_log_dir="s3://cr-qa-automation/logs/installer/release"
file=${file_prefix}_$timestamp.log
summary=${summary_prefix}_$timestamp.log
report=${report_prefix}_$timestamp.tar.gz

if [ $# -lt 1 ];then
	echo "error: missing version number"
	echo "usage: $0 0.1.0"
	exit 1
fi

#`curl https://2lczhzl702.execute-api.us-west-2.amazonaws.com/dev/build?version=$1\&timestamp=$timestamp > /dev/null 2>&1`

curl -H "Content-Type:application/json" -X POST -d "{\"image\":\"$image\",\"job\":\"$job\",\"params\":{\"version\":\"$1\",\"timestamp\":\"$timestamp\"}}" https://2lczhzl702.execute-api.us-west-2.amazonaws.com/dev/build > /dev/null 2>&1

#wait for s3 log ready
result=""
for ((i=0; i<20; i++))
do
	result=$(aws s3 ls ${s3_log_dir}/$file)
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
aws s3 cp ${s3_log_dir}/$file ./
cat $file

#wait for testing log ready
test_result=""
for ((i=0; i<30; i++))
do
        test_result=$(aws s3 ls ${s3_log_dir}/$summary)
        if [ "$test_result" != "" ];then
                echo "test log is generated."
                break
        fi

        echo "Testing is ongoing, pls wait......"
        sleep 30
done

if [ "$test_result" == "" ];then
        echo "No reponse in long time of testing, exit running, pls check it."
        exit 1
fi

#delay 5 seconds to avoiding it 's uploading
sleep 5
aws s3 cp ${s3_log_dir}/$summary ./
cat $summary

echo "Testing is done, you may check details by downloading file report.tar.gz from link ${s3_log_dir}/$report if you want it."
echo "terminate instance......"
instanceId=$(grep instanceId $file | cut -d= -f 2)
aws ec2 terminate-instances --instance-id $instanceId
