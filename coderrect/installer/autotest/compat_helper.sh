#/bin/bash

job=compat_test
image=ami-0e39a2aad2782dc20
summary_prefix=compat_summary
report_prefix=compat_log
timestamp=$(date +%s)

s3_log_dir="s3://cr-qa-automation/logs/compat-test"
summary=${summary_prefix}_$timestamp.log
report=${report_prefix}_$timestamp.tar.gz

if [ $# -lt 1 ];then
	echo "error: missing version number"
	echo "usage: $0 0.1.0"
	exit 1
fi

curl -H "Content-Type:application/json" -X POST -d "{\"image\":\"$image\",\"job\":\"$job\",\"params\":{\"version\":\"$1\",\"timestamp\":\"$timestamp\"}}" https://2lczhzl702.execute-api.us-west-2.amazonaws.com/dev/build > /dev/null 2>&1

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

echo "Testing is done, you may check details by downloading test logs from below link if you want it:"
echo "https://cr-qa-automation.s3-us-west-2.amazonaws.com/logs/compat-test/$report"
echo "terminate instance......"
instanceId=$(grep instanceId $summary | cut -d= -f 2)
aws ec2 terminate-instances --instance-id $instanceId
