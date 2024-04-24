#/bin/bash

s3_log_dir="s3://cr-qa-automation/logs/installer/master/"
file=build-installer-master.log

if [ $# -lt 1 ];then
	echo "error: missing version number"
	echo "usage: $0 0.1.0"
	exit 1
fi

aws s3 rm $s3_log_dir$file

`curl https://3lb9j0aafe.execute-api.us-west-2.amazonaws.com/dev/installer_master?version=$1 > /dev/null 2>&1`

#wait for s3 log ready
for ((i=0; i<50; i++))
do
	result=$(aws s3 ls $s3_log_dir | grep $file)
	if [ "$result" != "" ];then
		echo "Build log is generated."
        	break
        fi
	
	echo "The building is ongoing, pls wait......"
	sleep 30
done

aws s3 cp $s3_log_dir$file ./
cat $file

echo "terminate instance......"
instanceId=$(grep instanceId $file | cut -d= -f 2)
aws ec2 terminate-instances --instance-id $instanceId
